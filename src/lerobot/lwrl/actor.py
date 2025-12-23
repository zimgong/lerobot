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
import sys
import time
from functools import lru_cache
from queue import Empty

import grpc
import torch
from torch import nn
from torch.multiprocessing import Event, Queue

from lerobot.cameras import opencv  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.policies.factory import make_policy
from lerobot.policies.sac.modeling_sac import SACPolicy
from lerobot.policies.sac.modeling_flowrl import SACFlowRLPolicy as CurrentPolicy
from lerobot.processor import TransitionKey
from lerobot.robots import so100_follower  # noqa: F401
from lerobot.teleoperators import gamepad, so101_leader  # noqa: F401
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import (
    bytes_to_state_dict,
    grpc_channel_options,
    python_object_to_bytes,
    receive_bytes_in_chunks,
    send_bytes_in_chunks,
    transitions_to_bytes,
)
from lerobot.rl.process import ProcessSignalHandler
from lerobot.rl.queue import get_last_item_from_queue
from lerobot.utils.random_utils import set_seed
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.transition import (
    Transition,
    move_state_dict_to_device,
    move_transition_to_device,
)
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

from collections import deque
ACTOR_SHUTDOWN_TIMEOUT = 30

# Main entry point


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
        display = None
        if not has_display():
            logging.info("No display found, creating virtual display")
            import pyvirtualdisplay

            display = pyvirtualdisplay.Display(visible=0, size=(1920, 1080))
            display.start()
        else:
            logging.info("Display found, using existing display")
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
    observation_preprocessor = getattr(policy, "observation_preprocessor", None)

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

        observation = {}
        for k, v in transition[TransitionKey.OBSERVATION].items():
            if k in cfg.env.features_map:
                observation[cfg.env.features_map[k]] = v
            elif k in cfg.policy.input_features:
                observation[k] = v
        if observation_preprocessor:
            observation = observation_preprocessor(observation)

        if interaction_step >= cfg.policy.online_step_before_learning:
            # Time policy inference and check if it meets FPS requirement
            with policy_timer:
                # Extract observation from transition for policy
                action = policy.select_action(batch=observation)
                if action.isnan().any():
                    raise ValueError("Action is NaN")
            policy_fps = policy_timer.fps_last
            if observation_preprocessor:
                observation.pop("pixel_values")
                observation.pop("input_ids")
                observation.pop("attention_mask")
                observation.pop("position_ids")
                observation.pop("image_flags")
                observation.pop("labels")

            log_policy_frequency_issue(policy_fps=policy_fps, cfg=cfg, interaction_step=interaction_step)
        else:
            action = online_env.action_space.sample()

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
            
        list_transition_to_send_to_learner.append(
            Transition(
                state=observation,
                action=executed_action,
                reward=reward,
                next_state=next_observation,
                done=done,
                truncated=truncated,
                complementary_info={
                    "is_success": info.get('is_success', torch.zeros_like(done, device=device, dtype=torch.bool)).to(torch.float32),
                },
            )
        )
        # assign obs to the next obs and continue the rollout
        if torch.any(done) or torch.any(truncated):
            transition = new_transition_with_reset
        else:
            transition = new_transition

        if time.time() - last_time_policy_received > policy_parameters_push_frequency:
            logging.info(f"[ACTOR] Global step {interaction_step}: Running average reward: {sum(reward_running_buffer) / len(reward_running_buffer)}")
            update_policy_parameters(policy=policy, parameters_queue=parameters_queue, device=device)
            last_time_policy_received = time.time()

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
        

#  Communication Functions - Group all gRPC/messaging functions


def establish_learner_connection(
    stub: services_pb2_grpc.LearnerServiceStub,
    shutdown_event: Event,  # type: ignore
    attempts: int = 30,
):
    """Establish a connection with the learner.

    Args:
        stub (services_pb2_grpc.LearnerServiceStub): The stub to use for the connection.
        shutdown_event (Event): The event to check if the connection should be established.
        attempts (int): The number of attempts to establish the connection.
    Returns:
        bool: True if the connection is established, False otherwise.
    """
    for _ in range(attempts):
        if shutdown_event.is_set():
            logging.info("[ACTOR] Shutting down establish_learner_connection")
            return False

        # Force a connection attempt and check state
        try:
            logging.info("[ACTOR] Send ready message to Learner")
            if stub.Ready(services_pb2.Empty()) == services_pb2.Empty():
                return True
        except grpc.RpcError as e:
            logging.error(f"[ACTOR] Waiting for Learner to be ready... {e}")
            time.sleep(2)
    return False


@lru_cache(maxsize=1)
def learner_service_client(
    host: str = "127.0.0.1",
    port: int = 50051,
) -> tuple[services_pb2_grpc.LearnerServiceStub, grpc.Channel]:
    """
    Returns a client for the learner service.

    GRPC uses HTTP/2, which is a binary protocol and multiplexes requests over a single connection.
    So we need to create only one client and reuse it.
    """

    channel = grpc.insecure_channel(
        f"{host}:{port}",
        grpc_channel_options(),
    )
    stub = services_pb2_grpc.LearnerServiceStub(channel)
    logging.info("[ACTOR] Learner service client created")
    return stub, channel


def receive_policy(
    cfg: TrainRLServerPipelineConfig,
    parameters_queue: Queue,
    shutdown_event: Event,  # type: ignore
    learner_client: services_pb2_grpc.LearnerServiceStub | None = None,
    grpc_channel: grpc.Channel | None = None,
):
    """Receive parameters from the learner.

    Args:
        cfg (TrainRLServerPipelineConfig): The configuration for the actor.
        parameters_queue (Queue): The queue to receive the parameters.
        shutdown_event (Event): The event to check if the process should shutdown.
    """
    logging.info("[ACTOR] Start receiving parameters from the Learner")
    if not use_threads(cfg):
        # Create a process-specific log file
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"actor_receive_policy_{os.getpid()}.log")

        # Initialize logging with explicit log file
        init_logging(log_file=log_file, display_pid=True)
        logging.info("Actor receive policy process logging initialized")

        # Setup process handlers to handle shutdown signal
        # But use shutdown event from the main process
        _ = ProcessSignalHandler(use_threads=False, display_pid=True)

    if grpc_channel is None or learner_client is None:
        learner_client, grpc_channel = learner_service_client(
            host=cfg.policy.actor_learner_config.learner_host,
            port=cfg.policy.actor_learner_config.learner_port,
        )

    try:
        iterator = learner_client.StreamParameters(services_pb2.Empty())
        receive_bytes_in_chunks(
            iterator,
            parameters_queue,
            shutdown_event,
            log_prefix="[ACTOR] parameters",
        )

    except grpc.RpcError as e:
        logging.error(f"[ACTOR] gRPC error: {e}")

    if not use_threads(cfg):
        grpc_channel.close()
    logging.info("[ACTOR] Received policy loop stopped")


def send_transitions(
    cfg: TrainRLServerPipelineConfig,
    transitions_queue: Queue,
    shutdown_event: any,  # Event,
    learner_client: services_pb2_grpc.LearnerServiceStub | None = None,
    grpc_channel: grpc.Channel | None = None,
) -> services_pb2.Empty:
    """
    Sends transitions to the learner.

    This function continuously retrieves messages from the queue and processes:

    - Transition Data:
        - A batch of transitions (observation, action, reward, next observation) is collected.
        - Transitions are moved to the CPU and serialized using PyTorch.
        - The serialized data is wrapped in a `services_pb2.Transition` message and sent to the learner.
    """

    if not use_threads(cfg):
        # Create a process-specific log file
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"actor_transitions_{os.getpid()}.log")

        # Initialize logging with explicit log file
        init_logging(log_file=log_file, display_pid=True)
        logging.info("Actor transitions process logging initialized")

    if grpc_channel is None or learner_client is None:
        learner_client, grpc_channel = learner_service_client(
            host=cfg.policy.actor_learner_config.learner_host,
            port=cfg.policy.actor_learner_config.learner_port,
        )

    try:
        learner_client.SendTransitions(
            transitions_stream(
                shutdown_event, transitions_queue, cfg.policy.actor_learner_config.queue_get_timeout
            )
        )
    except grpc.RpcError as e:
        logging.error(f"[ACTOR] gRPC error: {e}")

    logging.info("[ACTOR] Finished streaming transitions")

    if not use_threads(cfg):
        grpc_channel.close()
    logging.info("[ACTOR] Transitions process stopped")


def send_interactions(
    cfg: TrainRLServerPipelineConfig,
    interactions_queue: Queue,
    shutdown_event: Event,  # type: ignore
    learner_client: services_pb2_grpc.LearnerServiceStub | None = None,
    grpc_channel: grpc.Channel | None = None,
) -> services_pb2.Empty:
    """
    Sends interactions to the learner.

    This function continuously retrieves messages from the queue and processes:

    - Interaction Messages:
        - Contains useful statistics about episodic rewards and policy timings.
        - The message is serialized using `pickle` and sent to the learner.
    """

    if not use_threads(cfg):
        # Create a process-specific log file
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"actor_interactions_{os.getpid()}.log")

        # Initialize logging with explicit log file
        init_logging(log_file=log_file, display_pid=True)
        logging.info("Actor interactions process logging initialized")

        # Setup process handlers to handle shutdown signal
        # But use shutdown event from the main process
        _ = ProcessSignalHandler(use_threads=False, display_pid=True)

    if grpc_channel is None or learner_client is None:
        learner_client, grpc_channel = learner_service_client(
            host=cfg.policy.actor_learner_config.learner_host,
            port=cfg.policy.actor_learner_config.learner_port,
        )

    try:
        learner_client.SendInteractions(
            interactions_stream(
                shutdown_event, interactions_queue, cfg.policy.actor_learner_config.queue_get_timeout
            )
        )
    except grpc.RpcError as e:
        logging.error(f"[ACTOR] gRPC error: {e}")

    logging.info("[ACTOR] Finished streaming interactions")

    if not use_threads(cfg):
        grpc_channel.close()
    logging.info("[ACTOR] Interactions process stopped")


def transitions_stream(shutdown_event: Event, transitions_queue: Queue, timeout: float) -> services_pb2.Empty:  # type: ignore
    while not shutdown_event.is_set():
        try:
            message = transitions_queue.get(block=True, timeout=timeout)
        except Empty:
            logging.debug("[ACTOR] Transition queue is empty")
            continue

        yield from send_bytes_in_chunks(
            message, services_pb2.Transition, log_prefix="[ACTOR] Send transitions"
        )

    return services_pb2.Empty()


def interactions_stream(
    shutdown_event: Event,
    interactions_queue: Queue,
    timeout: float,  # type: ignore
) -> services_pb2.Empty:
    while not shutdown_event.is_set():
        try:
            message = interactions_queue.get(block=True, timeout=timeout)
        except Empty:
            logging.debug("[ACTOR] Interaction queue is empty")
            continue

        yield from send_bytes_in_chunks(
            message,
            services_pb2.InteractionMessage,
            log_prefix="[ACTOR] Send interactions",
        )

    return services_pb2.Empty()


#  Policy functions


def update_policy_parameters(policy: CurrentPolicy, parameters_queue: Queue, device):
    bytes_state_dict = get_last_item_from_queue(parameters_queue, block=False)
    if bytes_state_dict is not None:
        logging.info("[ACTOR] Load new parameters from Learner.")
        state_dicts = bytes_to_state_dict(bytes_state_dict)

        # TODO: check encoder parameter synchronization possible issues:
        # 1. When shared_encoder=True, we're loading stale encoder params from actor's state_dict
        #    instead of the updated encoder params from critic (which is optimized separately)
        # 2. When freeze_vision_encoder=True, we waste bandwidth sending/loading frozen params
        # 3. Need to handle encoder params correctly for both actor and discrete_critic
        # Potential fixes:
        # - Send critic's encoder state when shared_encoder=True
        # - Skip encoder params entirely when freeze_vision_encoder=True
        # - Ensure discrete_critic gets correct encoder state (currently uses encoder_critic)

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


#  Utilities functions


def push_transitions_to_transport_queue(transitions: list, transitions_queue):
    """Send transitions to learner in smaller chunks to avoid network issues.

    Args:
        transitions: List of transitions to send
        message_queue: Queue to send messages to learner
        chunk_size: Size of each chunk to send
    """
    transition_to_send_to_learner = []
    for transition in transitions:
        tr = move_transition_to_device(transition=transition, device="cpu")
        for key, value in tr["state"].items():
            if torch.isnan(value).any():
                logging.warning(f"Found NaN values in transition {key}")

        transition_to_send_to_learner.append(tr)

    transitions_queue.put(transitions_to_bytes(transition_to_send_to_learner))


def get_frequency_stats(timer: TimerManager) -> dict[str, float]:
    """Get the frequency statistics of the policy.

    Args:
        timer (TimerManager): The timer with collected metrics.

    Returns:
        dict[str, float]: The frequency statistics of the policy.
    """
    stats = {}
    if timer.count > 1:
        avg_fps = timer.fps_avg
        p90_fps = timer.fps_percentile(90)
        logging.debug(f"[ACTOR] Average policy frame rate: {avg_fps}")
        logging.debug(f"[ACTOR] Policy frame rate 90th percentile: {p90_fps}")
        stats = {
            "Policy frequency [Hz]": avg_fps,
            "Policy frequency 90th-p [Hz]": p90_fps,
        }
    return stats


def log_policy_frequency_issue(policy_fps: float, cfg: TrainRLServerPipelineConfig, interaction_step: int):
    if policy_fps < cfg.env.fps:
        logging.warning(
            f"[ACTOR] Policy FPS {policy_fps:.1f} below required {cfg.env.fps} at step {interaction_step}"
        )


def use_threads(cfg: TrainRLServerPipelineConfig) -> bool:
    return cfg.policy.concurrency.actor == "threads"


def has_display() -> bool:
    """
    Detects if there is a monitor/display available on the machine.
    Checks both the DISPLAY environment variable and attempts to verify
    that the X server is actually accessible.
    Returns:
        True if a display is available, False otherwise.
    """
    # Check if DISPLAY environment variable is set
    display = os.environ.get("DISPLAY")
    if not display:
        logging.info("No DISPLAY environment variable set")
        return False

    # On Linux, verify we can actually connect to the X server
    if sys.platform == "linux":
        try:
            import subprocess
            # Use xdpyinfo to verify X server is accessible
            # This is a standard utility available on most Linux systems
            result = subprocess.run(
                ["xdpyinfo", "-display", display],
                capture_output=True,
                timeout=1,
            )
            if result.returncode == 0:
                logging.info(
                    f"Display {display} is available and accessible"
                )
                return True
            else:
                logging.info(
                    f"Display {display} is set but not accessible"
                )
                return False
        except (FileNotFoundError, subprocess.TimeoutExpired):
            # xdpyinfo not available or timed out,
            # try alternative check with Xlib
            try:
                from Xlib import display as xdisplay
                d = xdisplay.Display(display)
                d.close()
                logging.info(
                    f"Display {display} is available and accessible"
                )
                return True
            except (ImportError, Exception) as e:
                logging.info(
                    f"Display {display} is set but accessibility "
                    f"cannot be verified: {e}"
                )
                # If we can't verify, assume display exists if DISPLAY is set
                return True
        except Exception as e:
            logging.info(
                f"Error checking display availability: {e}, "
                f"assuming display exists"
            )
            # If we can't verify, assume display exists if DISPLAY is set
            return True

    # For non-Linux platforms, just check if DISPLAY is set
    logging.info(f"Display {display} is set (non-Linux platform)")
    return True


if __name__ == "__main__":
    actor_cli()
