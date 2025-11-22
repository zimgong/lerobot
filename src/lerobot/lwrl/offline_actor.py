#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Actor server runner for distributed HILSerl robot policy training.

This script implements the actor component of the distributed HILSerl architecture.
It executes the policy in the robot environment, collects experience,
and sends transitions to the learner server for policy updates.

Examples of usage:

- Start an actor server for real robot training with human-in-the-loop intervention:
```bash
python -m lerobot.lwrl.actor --config_path src/lerobot/configs/train_config_hilserl_so100.json
```

**NOTE**: The actor server requires a running learner server to connect to. Ensure the learner
server is started before launching the actor.

**NOTE**: Human intervention is key to HILSerl training. Press the upper right trigger button on the
gamepad to take control of the robot during training. Initially intervene frequently, then gradually
reduce interventions as the policy improves.

**WORKFLOW**:
1. Determine robot workspace bounds using `find_joint_limits.py`
2. Record demonstrations with `gym_manipulator.py` in record mode
3. Process the dataset and determine camera crops with `crop_dataset_roi.py`
4. Start the learner server with the training configuration
5. Start this actor server with the same configuration
6. Use human interventions to guide policy learning

For more details on the complete HILSerl training workflow, see:
https://github.com/michel-aractingi/lerobot-hilserl-guide
"""

import logging
import os
import time

import torch
from torch import nn, Tensor
from torch.multiprocessing import Event, Queue

from lerobot.cameras import opencv  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.policies.factory import make_policy
from lerobot.policies.sac.modeling_flowrl import SACFlowRLPolicy
from lerobot.policies.offline.modeling_offline import OfflineIQLPolicy as CurrentPolicy
from lerobot.processor import TransitionKey
from lerobot.robots import so100_follower  # noqa: F401
from lerobot.teleoperators import gamepad, so101_leader  # noqa: F401
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import (
    python_object_to_bytes,
)
from lerobot.rl.process import ProcessSignalHandler
from lerobot.utils.random_utils import set_seed
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.transition import Transition
from lerobot.utils.utils import (
    TimerManager,
    get_safe_torch_device,
    init_logging,
)

from lerobot.rl.gym_manipulator import (
    create_transition,
    make_processors,
    make_robot_env,
    step_env_and_process_transition,
)

# Import shared functions from actor.py
from lerobot.lwrl.actor import (
    ACTOR_SHUTDOWN_TIMEOUT,
    establish_learner_connection,
    get_frequency_stats,
    interactions_stream,
    learner_service_client,
    log_policy_frequency_issue,
    push_transitions_to_transport_queue,
    receive_policy,
    send_interactions,
    send_transitions,
    transitions_stream,
    update_policy_parameters,
    use_threads,
)

from collections import deque

# Main entry point
@torch.no_grad()
def _critic_stats(policy, obs: dict[str, Tensor], act: Tensor, obs_feats=None):
    """Compute Q_min and V for a given observation-action pair."""
    # Q (ensemble), take conservative Q_min for logging; and V(s)
    q_values = policy.critic_ensemble(obs, act, obs_feats)  # [n_heads, B]
    q_min = q_values.min(dim=0)[0]  # [B]
    v = policy.value_forward(obs, observation_features=obs_feats, use_target=False, detach_encoder=True)
    return q_min, v


@torch.no_grad()
def _actor_logprob(policy, obs: dict[str, Tensor], obs_feats=None, act: Tensor | None = None):
    """Get action and log-probability from actor policy."""
    dist, _ = policy._actor_distribution(obs, observation_features=obs_feats, detach_encoder=True)
    if act is None:
        act = dist.mode()
    return act, dist.log_prob(act)


@parser.wrap()
def actor_cli(cfg: TrainRLServerPipelineConfig):
    cfg.validate()
    display_pid = False
    if not use_threads(cfg):
        import torch.multiprocessing as mp

        mp.set_start_method("spawn")
        display_pid = True

    # Create logs directory to ensure it exists
    log_dir = os.path.join(cfg.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"actor_{cfg.job_name}.log")

    # Initialize logging with explicit log file
    init_logging(log_file=log_file, display_pid=display_pid)
    logging.info(f"Actor logging initialized, writing to {log_file}")

    is_threaded = use_threads(cfg)
    shutdown_event = ProcessSignalHandler(is_threaded, display_pid=display_pid).shutdown_event

    learner_client, grpc_channel = learner_service_client(
        host=cfg.policy.actor_learner_config.learner_host,
        port=cfg.policy.actor_learner_config.learner_port,
    )

    logging.info("[ACTOR] Establishing connection with Learner")
    if not establish_learner_connection(learner_client, shutdown_event):
        logging.error("[ACTOR] Failed to establish connection with Learner")
        return

    if not use_threads(cfg):
        # If we use multithreading, we can reuse the channel
        grpc_channel.close()
        grpc_channel = None

    logging.info("[ACTOR] Connection with Learner established")

    parameters_queue = Queue()
    transitions_queue = Queue()
    interactions_queue = Queue()

    concurrency_entity = None
    if use_threads(cfg):
        from threading import Thread

        concurrency_entity = Thread
    else:
        from multiprocessing import Process

        concurrency_entity = Process

    receive_policy_process = concurrency_entity(
        target=receive_policy,
        args=(cfg, parameters_queue, shutdown_event, grpc_channel),
        daemon=True,
    )

    transitions_process = concurrency_entity(
        target=send_transitions,
        args=(cfg, transitions_queue, shutdown_event, grpc_channel),
        daemon=True,
    )

    interactions_process = concurrency_entity(
        target=send_interactions,
        args=(cfg, interactions_queue, shutdown_event, grpc_channel),
        daemon=True,
    )

    transitions_process.start()
    interactions_process.start()
    receive_policy_process.start()

    # overwrite the policy with resume=False
    #! Lerobot from_pretrain take the same config_path arg with rl config, will cause error
    cfg.policy.resume = False
    cfg.policy.pretrained_path = None
    act_with_policy(
        cfg=cfg,
        shutdown_event=shutdown_event,
        parameters_queue=parameters_queue,
        transitions_queue=transitions_queue,
        interactions_queue=interactions_queue,
        policy_parameters_push_frequency=cfg.policy.actor_learner_config.policy_parameters_push_frequency,
    )
    logging.info("[ACTOR] Policy process joined")

    logging.info("[ACTOR] Closing queues")
    transitions_queue.close()
    interactions_queue.close()
    parameters_queue.close()

    transitions_process.join()
    logging.info("[ACTOR] Transitions process joined")
    interactions_process.join()
    logging.info("[ACTOR] Interactions process joined")
    receive_policy_process.join()
    logging.info("[ACTOR] Receive policy process joined")

    logging.info("[ACTOR] join queues")
    transitions_queue.cancel_join_thread()
    interactions_queue.cancel_join_thread()
    parameters_queue.cancel_join_thread()

    logging.info("[ACTOR] queues closed")


# Core algorithm functions


def act_with_policy(
    cfg: TrainRLServerPipelineConfig,
    shutdown_event: any,  # Event,
    parameters_queue: Queue,
    transitions_queue: Queue,
    interactions_queue: Queue,
    policy_parameters_push_frequency: int=15,
):
    """
    Executes policy interaction within the environment.

    This function rolls out the policy in the environment, collecting interaction data and pushing it to a queue for streaming to the learner.
    Once an episode is completed, updated network parameters received from the learner are retrieved from a queue and loaded into the network.

    Args:
        cfg: Configuration settings for the interaction process.
        shutdown_event: Event to check if the process should shutdown.
        parameters_queue: Queue to receive updated network parameters from the learner.
        transitions_queue: Queue to send transitions to the learner.
        interactions_queue: Queue to send interactions to the learner.
    """
    # Initialize logging for multiprocessing
    if not use_threads(cfg):
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"actor_policy_{os.getpid()}.log")
        init_logging(log_file=log_file, display_pid=True)
        logging.info("Actor policy process logging initialized")

    logging.info("make_env online")

    if cfg.env.type == "lwlab":
        from lerobot.lwrl.sim.lwlab.env_lwlab import make_lwlab_robot_env, make_lwlab_processors
        online_env, teleop_device = make_lwlab_robot_env(cfg=cfg.env)
        env_processor, action_processor = make_lwlab_processors(online_env, teleop_device, cfg.env, cfg.policy.device)
    else:
        online_env, teleop_device = make_robot_env(cfg=cfg.env)
        env_processor, action_processor = make_processors(online_env, teleop_device, cfg.env, cfg.policy.device)

    set_seed(cfg.seed)
    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("make_policy")

    ### Instantiate the policy in both the actor and learner processes
    ### To avoid sending a SACPolicy object through the port, we create a policy instance
    ### on both sides, the learner sends the updated parameters every n steps to update the actor's parameters
    policy: CurrentPolicy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
    )
    policy = policy.eval()
    assert isinstance(policy, nn.Module)

    obs, info = online_env.reset()
    env_processor.reset()
    action_processor.reset()

    # Process initial observation
    transition = create_transition(
        observation=obs,
        reward=torch.zeros((online_env.num_envs,), dtype=torch.float32, device=online_env.device),
        done=torch.zeros((online_env.num_envs,), dtype=torch.bool, device=online_env.device),
        truncated=torch.zeros((online_env.num_envs,), dtype=torch.bool, device=online_env.device),
        info=info)
    transition = env_processor(transition)

    # NOTE: For the moment we will solely handle the case of a single environment
    sum_reward_episode = 0
    list_transition_to_send_to_learner = []
    episode_intervention = False
    # Add counters for intervention rate calculation
    episode_intervention_steps = 0
    episode_total_steps = 0
    # for multi-env using running average reward
    reward_running_buffer = deque(maxlen=100)
    success_running_buffer = deque(maxlen=100)

    policy_timer = TimerManager("Policy inference", log=False)

    last_time_policy_received = time.time()

    for interaction_step in range(cfg.policy.online_steps):
        start_time = time.perf_counter()
        if shutdown_event.is_set():
            logging.info("[ACTOR] Shutting down act_with_policy")
            return

        observation = {
            k: v for k, v in transition[TransitionKey.OBSERVATION].items() if k in cfg.policy.input_features
        }

        # Time policy inference and check if it meets FPS requirement
        with policy_timer:
            # Extract observation from transition for policy
            # action = policy.select_action(batch=observation)

            # Optional cached features if the encoder is frozen
            obs_feats = None
            if getattr(policy.config, "freeze_vision_encoder", False) and policy.actor.encoder.has_images:
                obs_feats = policy.actor.encoder.get_cached_image_features(observation)

            action, logp = _actor_logprob(policy, observation, obs_feats)
            if action.isnan().any():
                raise ValueError("Action is NaN")
        policy_fps = policy_timer.fps_last

        log_policy_frequency_issue(policy_fps=policy_fps, cfg=cfg, interaction_step=interaction_step)

        # Use the new step function
        if cfg.env.type == "lwlab":
            from lerobot.lwrl.sim.lwlab.env_lwlab import step_lwlab_env_and_process_transition
            new_transition = step_lwlab_env_and_process_transition(
                env=online_env,
                transition=transition,
                action=action,
                env_processor=env_processor,
                action_processor=action_processor,
            )
        else:
            new_transition = step_env_and_process_transition(
                env=online_env,
                transition=transition,
                action=action,
                env_processor=env_processor,
                action_processor=action_processor,
            )

        # Extract values from processed transition
        next_observation = {
            k: v
            for k, v in new_transition[TransitionKey.OBSERVATION].items()
            if k in cfg.policy.input_features
        }

        # Teleop action is the action that was executed in the environment
        # It is either the action from the teleop device or the action from the policy
        executed_action = new_transition[TransitionKey.ACTION]
        reward = new_transition[TransitionKey.REWARD]
        done = new_transition.get(TransitionKey.DONE, torch.tensor(False))
        truncated = new_transition.get(TransitionKey.TRUNCATED, torch.tensor(False))
        info = new_transition.get(TransitionKey.INFO, {})

        # Compute Q_min(s,a) and V(s) for OPE
        q_min, v = _critic_stats(policy, observation, executed_action, obs_feats)
        # Compute advantage for diagnostics
        adv = q_min - v

        sum_reward_episode += float(reward.mean())
        # Increment total steps counter for intervention rate
        episode_total_steps += 1
        # for multi_env
        reward_running_buffer.append(reward.mean())
        # insert 0 / 1 to success_running_buffer
        num_success = info.get('is_success', torch.zeros_like(done, device=device, dtype=torch.bool)).sum().item()
        num_failure = torch.logical_or(done, truncated).sum().item() - num_success
        for _ in range(num_success):
            success_running_buffer.append(1)
        for _ in range(num_failure):
            success_running_buffer.append(0)

        #! Handle IsaacSim Lwlab Last timestamp Bug, the env will be automatically reset
        #! so need to manually replace next_obs with info['final_obs']
        if torch.any(done) or torch.any(truncated):
            new_transition_with_reset = new_transition
            # re-write done and truncated
            new_transition_with_reset[TransitionKey.DONE] = torch.zeros_like(done, device=device, dtype=torch.bool)
            new_transition_with_reset[TransitionKey.TRUNCATED] = torch.zeros_like(truncated, device=device, dtype=torch.bool)
            new_transition_with_reset[TransitionKey.REWARD] = torch.zeros_like(reward, device=device, dtype=torch.float32)
            new_transition_with_reset[TransitionKey.INFO] = {}
            #! original code will reset processor here, but skip here
            # TODO: need to implement reset processor per env index
            # env_processor.reset()
            # action_processor.reset()
            
            # recreate real transition and overwrite next observation (pass processer)
            next_observation_raw = info['final_obs']['policy'] # replace with last obs before reset
            new_transition_raw = create_transition(
                observation=next_observation_raw, info=info,
                done=torch.zeros_like(done, device=device, dtype=torch.bool),
                truncated=torch.zeros_like(truncated, device=device, dtype=torch.bool),
                reward=torch.zeros_like(reward, device=device, dtype=torch.float32),
            )
            # Extract values from processed transition
            new_transition = env_processor(new_transition_raw)
            next_observation = {
                k: v
                for k, v in new_transition[TransitionKey.OBSERVATION].items()
                if k in cfg.policy.input_features
            }
            # make sure those will not be used!! (only create to use processer)
            del new_transition, new_transition_raw

            info.pop('final_obs') # remove final_obs from info to save space

        complementary_info = {
            "log_prob_beh": logp.cpu(),  # for PPO-style ratio
            "q_min": q_min.cpu(),  # AM-Q
            "v": v.cpu(),  # optional diagnostics
            "adv": adv.cpu(),  # optional diagnostics
            "is_success": info.get('is_success', torch.zeros_like(done, device=device, dtype=torch.bool)).to(torch.float32),
        }
            
        list_transition_to_send_to_learner.append(
            Transition(
                state=observation,
                action=executed_action,
                reward=reward,
                next_state=next_observation,
                done=done,
                truncated=truncated,
                complementary_info=complementary_info,
            )
        )
        # assign obs to the next obs and continue the rollout
        if torch.any(done) or torch.any(truncated):
            transition = new_transition_with_reset
        else:
            transition = new_transition

        # Check if new parameters are available in queue (non-blocking check)
        has_new_params = check_new_parameters_available(parameters_queue)
        
        if has_new_params:
            # Try to fetch new parameters with retry: every 10s for up to 60s
            params_updated = False
            for retry in range(6):  # 6 retries * 10s = 60s max
                if update_policy_parameters_with_retry(policy=policy, parameters_queue=parameters_queue, device=device):
                    params_updated = True
                    last_time_policy_received = time.time()
                    logging.info(f"[ACTOR] Policy parameters updated at step {interaction_step}")
                    break
                if retry < 6:  # Don't sleep on last retry
                    time.sleep(10)
            
            if not params_updated:
                logging.warning(f"[ACTOR] Failed to fetch new parameters after 60s at step {interaction_step}")

        # Periodically log stats and send interactions (based on time)
        if time.time() - last_time_policy_received > 10:
            logging.info(f"[ACTOR] Global step {interaction_step}: Running average reward: {sum(reward_running_buffer) / len(reward_running_buffer)}")
            
            stats = get_frequency_stats(policy_timer)
            policy_timer.reset()

            # Send episodic reward to the learner
            interactions_queue.put(
                python_object_to_bytes(
                    {
                        "Interaction step": interaction_step,
                        "Running average reward": float(sum(reward_running_buffer) / (len(reward_running_buffer) + 1e-10)),
                        "Success rate": float(sum(success_running_buffer) / (len(success_running_buffer) + 1e-10)),
                        **stats,
                    }
                )
            )
        
        # Send transitions to the learner
        # TODO: sz: check blocking time when sent every time
        if len(list_transition_to_send_to_learner) > 0:
            push_transitions_to_transport_queue(
                transitions=list_transition_to_send_to_learner,
                transitions_queue=transitions_queue,
            )
            list_transition_to_send_to_learner = []

        if cfg.env.fps is not None and cfg.env.type not in {"lwlab"}:
            dt_time = time.perf_counter() - start_time
            busy_wait(1 / cfg.env.fps - dt_time)

        # # manually save the policy and checkpoint
        # # TODO: (sz) need to fix the saving issue in learner
        # if interaction_step % cfg.save_freq == 0:
        #     import pickle
        #     from lerobot.utils.train_utils import get_step_checkpoint_dir
        #     from lerobot.utils.constants import PRETRAINED_MODEL_DIR
        #     from pathlib import Path
            
        #     # save checkpoint only
        #     checkpoint_dir = get_step_checkpoint_dir(Path(cfg.output_dir) / "actor", interaction_step, interaction_step)
        #     pretrained_dir = checkpoint_dir / PRETRAINED_MODEL_DIR
        #     os.makedirs(pretrained_dir, exist_ok=True)
        #     policy.save_pretrained(pretrained_dir)
        #     cfg.save_pretrained(pretrained_dir)

        #     # pickle the policy
        #     with open(os.path.join(checkpoint_dir, f"policy_{interaction_step}.pkl"), "wb") as f:
        #         pickle.dump(policy, f)
        #     logging.info(f"[ACTOR] Saved policy and checkpoint at interaction step {interaction_step}")
        

# Communication and utility functions are imported from actor.py


def check_new_parameters_available(parameters_queue: Queue) -> bool:
    """Lightweight check if new parameters are available in queue (without draining)."""
    from queue import Empty
    
    # Quick non-blocking check if queue has items
    # Peek by getting and putting back (lightweight operation)
    try:
        item = parameters_queue.get_nowait()
        # Put it back so we can fetch it properly later with get_last_item_from_queue
        parameters_queue.put(item)
        return True
    except Empty:
        return False


def update_policy_parameters_with_retry(policy, parameters_queue: Queue, device) -> bool:
    """Try to update policy parameters from queue. Returns True if successful, False otherwise."""
    from lerobot.rl.queue import get_last_item_from_queue
    from lerobot.transport.utils import bytes_to_state_dict
    from lerobot.utils.transition import move_state_dict_to_device
    
    bytes_state_dict = get_last_item_from_queue(parameters_queue, block=False)
    if bytes_state_dict is not None:
        logging.info("[ACTOR] Load new parameters from Learner.")
        state_dicts = bytes_to_state_dict(bytes_state_dict)

        # Load actor state dict
        actor_state_dict = move_state_dict_to_device(state_dicts["policy"], device=device)
        policy.actor.load_state_dict(actor_state_dict)

        # Load discrete critic if present
        if hasattr(policy, "discrete_critic") and "discrete_critic" in state_dicts:
            discrete_critic_state_dict = move_state_dict_to_device(
                state_dicts["discrete_critic"], device=device
            )
            policy.discrete_critic.load_state_dict(discrete_critic_state_dict)
            logging.info("[ACTOR] Loaded discrete critic parameters from Learner.")
        
        return True
    return False


if __name__ == "__main__":
    actor_cli()
