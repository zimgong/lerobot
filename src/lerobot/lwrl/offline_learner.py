import logging
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from pprint import pformat
from tqdm import tqdm

import grpc
import torch
from torch.nn.utils import clip_grad_norm_
from termcolor import colored
from torch import nn
from torch.multiprocessing import Queue
from torch.optim.optimizer import Optimizer

from lerobot.cameras import opencv  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy
from lerobot.policies.sac.modeling_sac import SACPolicy # as CurrentPolicy
from lerobot.policies.sac.modeling_flowrl import SACFlowRLPolicy
from lerobot.policies.offline.modeling_offline import OfflineIQLPolicy as CurrentPolicy
from lerobot.rl.buffer import ReplayBuffer, concatenate_batch_transitions
from lerobot.rl.process import ProcessSignalHandler
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.robots import so100_follower  # noqa: F401
from lerobot.teleoperators import gamepad, so101_leader  # noqa: F401
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.transport import services_pb2_grpc
from lerobot.transport.utils import (
    MAX_MESSAGE_SIZE,
    bytes_to_python_object,
    bytes_to_transitions,
    state_to_bytes,
)
from lerobot.utils.constants import (
    ACTION,
    CHECKPOINTS_DIR,
    LAST_CHECKPOINT_LINK,
    PRETRAINED_MODEL_DIR,
    TRAINING_STATE_DIR,
)
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    load_training_state as utils_load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.transition import move_state_dict_to_device, move_transition_to_device
from lerobot.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    init_logging,
)

from lerobot.rl.learner_service import MAX_WORKERS, SHUTDOWN_TIMEOUT, LearnerService
from lerobot.lwrl.buffer_batched import ParallelReplayBuffer
from lerobot.lwrl.ope import amq_score_from_buffer, amq_score_from_buffer_sample, amq_score_from_buffer_online
from lerobot.lwrl.buffer_utils import merge_offline_online_success

# Import functions called BY add_actor_information_and_train from learner.py
from lerobot.lwrl.learner import (
    use_threads,
    push_actor_policy_to_queue,
    load_training_state,
    log_training_info,
    process_transitions,
    process_interaction_messages,
    check_nan_in_transition,
    get_observation_features,
    save_training_checkpoint,
    handle_resume_logic,
    initialize_replay_buffer, 
    initialize_offline_replay_buffer,

)


@parser.wrap()
def train_cli(cfg: TrainRLServerPipelineConfig):
    if not use_threads(cfg):
        import torch.multiprocessing as mp

        mp.set_start_method("spawn")

    # Use the job_name from the config
    train(
        cfg,
        job_name=cfg.job_name,
    )

    logging.info("[LEARNER] train_cli finished")


def train(cfg: TrainRLServerPipelineConfig, job_name: str | None = None):
    """
    Main training function that initializes and runs the training process.

    Args:
        cfg (TrainRLServerPipelineConfig): The training configuration
        job_name (str | None, optional): Job name for logging. Defaults to None.
    """

    cfg.validate()

    if job_name is None:
        job_name = cfg.job_name

    if job_name is None:
        raise ValueError("Job name must be specified either in config or as a parameter")

    display_pid = False
    if not use_threads(cfg):
        display_pid = True

    # Create logs directory to ensure it exists
    log_dir = os.path.join(cfg.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"learner_{job_name}.log")

    # Initialize logging with explicit log file
    init_logging(log_file=log_file, display_pid=display_pid)
    logging.info(f"Learner logging initialized, writing to {log_file}")
    logging.info(pformat(cfg.to_dict()))

    # Setup WandB logging if enabled
    if cfg.wandb.enable and cfg.wandb.project:
        from lerobot.rl.wandb_utils import WandBLogger

        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    # Handle resume logic
    cfg = handle_resume_logic(cfg)

    set_seed(seed=cfg.seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    is_threaded = use_threads(cfg)
    shutdown_event = ProcessSignalHandler(is_threaded, display_pid=display_pid).shutdown_event

    start_learner_threads(
        cfg=cfg,
        wandb_logger=wandb_logger,
        shutdown_event=shutdown_event,
    )


def start_learner_threads(
    cfg: TrainRLServerPipelineConfig,
    wandb_logger: WandBLogger | None,
    shutdown_event: any,  # Event,
) -> None:
    """
    Start the learner threads for training.

    Args:
        cfg (TrainRLServerPipelineConfig): Training configuration
        wandb_logger (WandBLogger | None): Logger for metrics
        shutdown_event: Event to signal shutdown
    """
    # Create multiprocessing queues
    transition_queue = Queue()
    interaction_message_queue = Queue()
    parameters_queue = Queue()

    concurrency_entity = None

    if use_threads(cfg):
        from threading import Thread

        concurrency_entity = Thread
    else:
        from torch.multiprocessing import Process

        concurrency_entity = Process

    communication_process = concurrency_entity(
        target=start_learner,
        args=(
            parameters_queue,
            transition_queue,
            interaction_message_queue,
            shutdown_event,
            cfg,
        ),
        daemon=True,
    )
    communication_process.start()

    add_actor_information_and_train(
        cfg=cfg,
        wandb_logger=wandb_logger,
        shutdown_event=shutdown_event,
        transition_queue=transition_queue,
        interaction_message_queue=interaction_message_queue,
        parameters_queue=parameters_queue,
    )
    logging.info("[LEARNER] Training process stopped")

    logging.info("[LEARNER] Closing queues")
    transition_queue.close()
    interaction_message_queue.close()
    parameters_queue.close()

    communication_process.join()
    logging.info("[LEARNER] Communication process joined")

    logging.info("[LEARNER] join queues")
    transition_queue.cancel_join_thread()
    interaction_message_queue.cancel_join_thread()
    parameters_queue.cancel_join_thread()

    logging.info("[LEARNER] queues closed")


# Core algorithm functions


def add_actor_information_and_train(
    cfg: TrainRLServerPipelineConfig,
    wandb_logger: WandBLogger | None,
    shutdown_event: any,  # Event,
    transition_queue: Queue,
    interaction_message_queue: Queue,
    parameters_queue: Queue,
):
    """
    Handles data transfer from the actor to the learner, manages training updates,
    and logs training progress in an online reinforcement learning setup.

    This function continuously:
    - Transfers transitions from the actor to the replay buffer.
    - Logs received interaction messages.
    - Ensures training begins only when the replay buffer has a sufficient number of transitions.
    - Samples batches from the replay buffer and performs multiple critic updates.
    - Periodically updates the actor, critic, and temperature optimizers.
    - Logs training statistics, including loss values and optimization frequency.

    NOTE: This function doesn't have a single responsibility, it should be split into multiple functions
    in the future. The reason why we did that is the  GIL in Python. It's super slow the performance
    are divided by 200. So we need to have a single thread that does all the work.

    Args:
        cfg (TrainRLServerPipelineConfig): Configuration object containing hyperparameters.
        wandb_logger (WandBLogger | None): Logger for tracking training progress.
        shutdown_event (Event): Event to signal shutdown.
        transition_queue (Queue): Queue for receiving transitions from the actor.
        interaction_message_queue (Queue): Queue for receiving interaction messages from the actor.
        parameters_queue (Queue): Queue for sending policy parameters to the actor.
    """
    # Extract all configuration variables at the beginning, it improve the speed performance
    # of 7%
    device = get_safe_torch_device(try_device=cfg.learner_device, log=True)
    storage_device = get_safe_torch_device(try_device=cfg.policy.storage_device)
    clip_grad_norm_value = cfg.policy.grad_clip_norm
    fps = cfg.env.fps
    log_freq = cfg.log_freq
    save_freq = cfg.save_freq
    policy_update_freq = cfg.policy.policy_update_freq
    saving_checkpoint = cfg.save_checkpoint
    async_prefetch = cfg.policy.async_prefetch

    # offline training parameters
    offline_iters = cfg.offline.iters
    iql_steps_per_iter = cfg.offline.iql_steps
    bc_steps_after_merge = cfg.offline.bc_steps_after_merge
    bc_warmup_steps = cfg.offline.bc_steps_after_merge
    ope_adaptive_threshold_fraction = cfg.offline.ope_adaptive_threshold_fraction
    max_ope_iterations = cfg.offline.max_ope_iterations
    ope_iterations = 0
    max_online_episodes_added_per_iter = cfg.offline.max_online_episodes_added_per_iter


    # build ope eval env
    ope_eval_env_cfg = cfg.ope_eval_env
    assert ope_eval_env_cfg.type == "lwlab", "LwLab environment must be provided"
    from lerobot.lwrl.sim.lwlab.env_lwlab import make_lwlab_robot_env, make_lwlab_processors
    ope_eval_env, ope_eval_teleop_device = make_lwlab_robot_env(cfg=ope_eval_env_cfg)
    ope_eval_env_processor, ope_eval_action_processor = make_lwlab_processors(env=ope_eval_env, teleop_device=ope_eval_teleop_device, cfg=ope_eval_env_cfg, device=device)

    # Initialize logging for multiprocessing
    if not use_threads(cfg):
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"learner_train_process_{os.getpid()}.log")
        init_logging(log_file=log_file, display_pid=True)
        logging.info("Initialized logging for actor information and training process")

    logging.info("Initializing policy")

    cfg.policy.device = cfg.learner_device # override for offline training
    policy: CurrentPolicy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
    )

    assert isinstance(policy, nn.Module)

    policy.train()

    push_actor_policy_to_queue(parameters_queue=parameters_queue, policy=policy)

    optimizers, lr_scheduler = make_optimizers_and_scheduler(cfg=cfg, policy=policy)

    # If we are resuming, we need to load the training state
    resume_optimization_step, resume_interaction_step = load_training_state(cfg=cfg, optimizers=optimizers)

    log_training_info(cfg=cfg, policy=policy)

    replay_buffer = initialize_replay_buffer(cfg, device, storage_device)
    batch_size = cfg.batch_size
    offline_replay_buffer = None

    if cfg.dataset is not None:
        offline_replay_buffer, allowed_features = initialize_offline_replay_buffer(
            cfg=cfg,
            device=device,
            storage_device=storage_device,
            return_features=True,
        )
        batch_size: int = batch_size # // 2  # We will sample from both replay buffer
    else:
        raise ValueError("Dataset is required for offline training")

    logging.info("Starting learner thread")
    interaction_message = None
    optimization_step = resume_optimization_step if resume_optimization_step is not None else 0
    interaction_step_shift = resume_interaction_step if resume_interaction_step is not None else 0

    dataset_repo_id = None
    if cfg.dataset is not None:
        dataset_repo_id = cfg.dataset.repo_id

    # Initialize iterators
    online_iterator = None
    offline_iterator = None

    # if resume_optimization_step is not None:
    #     progress_bar.update(resume_optimization_step)
    #     last_accept_step = resume_optimization_step

    # bc warm up

    if bc_warmup_steps > 0:
        logging.info(f"[OFFLINE] Running BC warmup for {bc_warmup_steps} steps")

        if online_iterator is None:
            online_iterator = replay_buffer.get_iterator(
                batch_size=batch_size, async_prefetch=async_prefetch, queue_size=2
            )
        
        if offline_replay_buffer is not None and offline_iterator is None:
            offline_iterator = offline_replay_buffer.get_iterator(
                batch_size=batch_size, async_prefetch=async_prefetch, queue_size=2
            )

        original_encoder_requires_grad = None
        if hasattr(policy.actor.encoder, "image_encoder") and len(list(policy.actor.encoder.image_encoder.parameters())) > 0:
            original_encoder_requires_grad = next(policy.actor.encoder.image_encoder.parameters()).requires_grad
        
        # Ensure actor encoder is trainable
        policy._ensure_actor_encoder_trainable()

        for bc_step in tqdm(range(bc_warmup_steps), desc="BC warmup"):

            # Process all available transitions to the replay buffer, send by the actor server
            process_transitions(
                transition_queue=transition_queue,
                replay_buffer=replay_buffer,
                offline_replay_buffer=offline_replay_buffer,
                device=storage_device,
                dataset_repo_id=dataset_repo_id,
                shutdown_event=shutdown_event,
            )

            # Process all available interaction messages sent by the actor server
            interaction_message = process_interaction_messages(
                interaction_message_queue=interaction_message_queue,
                interaction_step_shift=interaction_step_shift,
                wandb_logger=wandb_logger,
                shutdown_event=shutdown_event,
            )

            offline_batch = next(offline_iterator)
            online_batch = next(online_iterator)
            batch = concatenate_batch_transitions(
                left_batch_transitions=offline_batch, right_batch_transition=online_batch
            )

            actions = batch[ACTION]
            observations = batch["state"]
            next_observations = batch["next_state"]
            check_nan_in_transition(observations=observations, actions=actions, next_state=next_observations)

            observation_features, _ = get_observation_features(
                policy=policy, observations=observations, next_observations=next_observations
            )

            forward_batch = {
                ACTION: actions,
                "state": observations,
                "observation_feature": observation_features,
            }

            bc_out = policy.forward(forward_batch, model="actor_bc")
            optimizers["actor"].zero_grad()
            bc_out["loss_actor_bc"].backward()
            clip_grad_norm_(policy.actor.parameters(), clip_grad_norm_value)
            optimizers["actor"].step()

            if wandb_logger is not None and bc_step % log_freq == 0:
                wandb_logger.log_dict(
                    {"bc_loss": bc_out["loss_actor_bc"].item(), "BC step": bc_step},
                    mode="train",
                    custom_step_key="BC step",
                )
        
        # Reset encoder requires_grad to original value if it was set
        if original_encoder_requires_grad is not None:
            for param in policy.actor.encoder.image_encoder.parameters():
                param.requires_grad_(original_encoder_requires_grad)
        
        # Optionally sync encoders if not shared
        if cfg.offline.sync_critic_encoder_after_bc and not policy.shared_encoder:
            logging.info("[OFFLINE] Syncing critic encoder with actor encoder after BC...")
            policy.encoder_critic.load_state_dict(policy.actor.encoder.state_dict(), strict=False)
    
    # NOTE: THIS IS THE MAIN LOOP OF THE LEARNER
    # Outer loop: iterative offline stage
    for it in range(offline_iters):
        ope_iterations += 1

        logging.info(f"[OFFLINE] Starting iteration {it+1}/{offline_iters}")
        progress_bar = tqdm(range(iql_steps_per_iter), desc=f"Offline RL / IQL iter {it+1}/{offline_iters}")

        for iter_step in progress_bar:
            # Exit the training loop if shutdown is requested
            if shutdown_event is not None and shutdown_event.is_set():
                logging.info("[LEARNER] Shutdown signal received. Exiting...")
                break

            # Process all available transitions to the replay buffer, send by the actor server
            process_transitions(
                transition_queue=transition_queue,
                replay_buffer=replay_buffer,
                offline_replay_buffer=offline_replay_buffer,
                device=storage_device,
                dataset_repo_id=dataset_repo_id,
                shutdown_event=shutdown_event,
            )

            # Process all available interaction messages sent by the actor server
            interaction_message = process_interaction_messages(
                interaction_message_queue=interaction_message_queue,
                interaction_step_shift=interaction_step_shift,
                wandb_logger=wandb_logger,
                shutdown_event=shutdown_event,
            )

            if offline_replay_buffer is not None and offline_iterator is None:
                offline_iterator = offline_replay_buffer.get_iterator(
                    batch_size=batch_size, async_prefetch=async_prefetch, queue_size=2
                )

            time_for_one_optimization_step = time.time()
            
            # Sample for the last update in the UTD ratio
            batch = next(offline_iterator)

            actions = batch[ACTION]
            rewards = batch["reward"]
            observations = batch["state"]
            next_observations = batch["next_state"]
            done = batch["done"]

            check_nan_in_transition(observations=observations, actions=actions, next_state=next_observations)

            observation_features, next_observation_features = get_observation_features(
                policy=policy, observations=observations, next_observations=next_observations
            )

            # Create a batch dictionary with all required elements for the forward method
            forward_batch = {
                ACTION: actions,
                "reward": rewards,
                "state": observations,
                "next_state": next_observations,
                "done": done,
                "observation_feature": observation_features,
                "next_observation_feature": next_observation_features,
            }

            forward_batch["observation_feature"] = observation_features
            forward_batch["next_observation_feature"] = next_observation_features

            # Initialize training info dictionary
            training_infos = {}

            # ------------- VALUE (expectile) -------------
            value_output = policy.forward(forward_batch, model="value")
            loss_value = value_output["loss_value"]
            optimizers["value"].zero_grad()
            loss_value.backward()
            value_grad_norm = clip_grad_norm_(policy.value_head.parameters(), clip_grad_norm_value).item()
            optimizers["value"].step()

            # ------------- CRITIC (TD with V bootstrap) -------------
            critic_output = policy.forward(forward_batch, model="critic")
            loss_critic = critic_output["loss_critic"]
            optimizers["critic"].zero_grad()
            loss_critic.backward()
            critic_grad_norm = clip_grad_norm_(policy.critic_ensemble.parameters(), clip_grad_norm_value).item()
            optimizers["critic"].step()

            # ------------- ACTOR (AWR or PPO-style) -------------
            if cfg.policy.actor_update == "awr":
                actor_output = policy.forward(forward_batch, model="actor")
                loss_actor = actor_output["loss_actor"]
                do_update = True
            else:
                # PPO-style (needs log_prob_beh from complementary_info)
                actor_output = policy.compute_loss_actor_ppo(forward_batch)
                loss_actor = actor_output["loss_actor_ppo"]
                do_update = True

            actor_grad_norm = None
            if do_update and ((optimization_step + 1) % policy_update_freq == 0):
                optimizers["actor"].zero_grad()
                loss_actor.backward()
                actor_grad_norm = clip_grad_norm_(
                    [p for n, p in policy.actor.named_parameters() if not policy.shared_encoder or not n.startswith("encoder")],
                    clip_grad_norm_value,
                ).item()
                optimizers["actor"].step()

            # Discrete critic optimization (if available)
            if policy.config.num_discrete_actions is not None:
                discrete_critic_output = policy.forward(forward_batch, model="discrete_critic")
                loss_discrete_critic = discrete_critic_output["loss_discrete_critic"]
                optimizers["discrete_critic"].zero_grad()
                loss_discrete_critic.backward()
                discrete_critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                    parameters=policy.discrete_critic.parameters(), max_norm=clip_grad_norm_value
                ).item()
                optimizers["discrete_critic"].step()

                # Add discrete critic info to training info
                training_infos["loss_discrete_critic"] = loss_discrete_critic.item()
                training_infos["discrete_critic_grad_norm"] = discrete_critic_grad_norm
                training_infos.update(discrete_critic_output.get("q_info", {}))

            policy.update_target_networks()

            # logging
            training_infos.update({
                "loss_value": loss_value.item(),
                "loss_critic": loss_critic.item(),
                "loss_actor": loss_actor.item(),
                "value_grad_norm": value_grad_norm,
                "critic_grad_norm": critic_grad_norm,
            })

            if actor_grad_norm is not None:
                training_infos["actor_grad_norm"] = actor_grad_norm

            training_infos.update(critic_output.get("q_info", {}))
            training_infos.update(actor_output.get("actor_info", {}))
            if "ratio_mean" in actor_output:
                training_infos["ratio_mean"] = actor_output["ratio_mean"].item()
            training_infos["Training step"] = optimization_step + 1

            progress_bar.set_postfix(
                {
                    "loss_c": f"{training_infos['loss_critic']:.3f}",
                    "loss_v": f"{training_infos['loss_value']:.3f}",
                    "loss_a": f"{training_infos['loss_actor']:.3f}",
                }
            )

            if lr_scheduler is not None: lr_scheduler.step()

            # Log training metrics at specified intervals
            if optimization_step % log_freq == 0:
                training_infos["replay_buffer_size"] = len(replay_buffer)
                if offline_replay_buffer is not None:
                    training_infos["offline_replay_buffer_size"] = len(offline_replay_buffer)
                training_infos["Optimization step"] = optimization_step

                # Log training metrics
                if wandb_logger:
                    wandb_logger.log_dict(d=training_infos, mode="train", custom_step_key="Optimization step")

            # Calculate and log optimization frequency
            time_for_one_optimization_step = time.time() - time_for_one_optimization_step
            frequency_for_one_optimization_step = 1 / (time_for_one_optimization_step + 1e-9)

            # Log optimization frequency
            if wandb_logger:
                wandb_logger.log_dict(
                    {
                        "Optimization frequency loop [Hz]": frequency_for_one_optimization_step,
                        "Optimization step": optimization_step,
                    },
                    mode="train",
                    custom_step_key="Optimization step",
                )

            optimization_step += 1
            if optimization_step % log_freq == 0:
                progress_bar.update(log_freq)

            # Save checkpoint at specified intervals
            if saving_checkpoint and (optimization_step % save_freq == 0 or optimization_step == iql_steps_per_iter * offline_iters):
                save_training_checkpoint(
                    cfg=cfg,
                    optimization_step=optimization_step,
                    online_steps=iql_steps_per_iter * offline_iters,
                    interaction_message=interaction_message,
                    policy=policy,
                    optimizers=optimizers,
                    replay_buffer=replay_buffer,
                    offline_replay_buffer=offline_replay_buffer,
                    dataset_repo_id=dataset_repo_id,
                    fps=fps,
                )
        
        # --------- OPE gate at end of iteration ----------
        logging.info("[OFFLINE] Evaluating policy with OPE (AM-Q)...")
        # cand_score, cand_frames = amq_score_from_buffer(replay_buffer)
        # Use online evaluation: sample from buffer and compute Q-values with current policy
        ope_num_samples = getattr(cfg.offline, "ope_num_samples", min(5000, len(replay_buffer)))
        cand_score, cand_frames, improvement, update_policy = amq_score_from_buffer_online(
            buf=replay_buffer,
            policy=policy,
            cfg=cfg,
            num_samples=ope_num_samples,
            adaptive_threshold_fraction=getattr(cfg.offline, "ope_adaptive_threshold_fraction", 0.05),
            online_env=ope_eval_env,
            online_env_processor=ope_eval_env_processor,
            online_action_processor=ope_eval_action_processor,
        )

        logging.info(f"[OFFLINE] OPE score: {cand_score:.2f}, samples: {cand_frames}, improvement: {improvement:.3f}, threshold_fraction: {ope_adaptive_threshold_fraction}")

        # Note: ope_min_episodes is now interpreted as minimum frames for acceptance gate
        if update_policy or ope_iterations >= max_ope_iterations:
            ope_iterations = 0
            
            # Accept and refresh reference score
            logging.info(f"[OFFLINE] Policy accepted! Current Score: {cand_score:.2f}")

            # Merge offline dataset with SUCCESSFUL online episodes
            logging.info("[OFFLINE] Merging successful online episodes into offline buffer...")

            offline_replay_buffer = merge_offline_online_success(
                offline_buffer=offline_replay_buffer,
                online_buffer=replay_buffer,
                allowed_features=allowed_features,
                task_name=cfg.env.task if cfg.env.task is not None else "Control robot to finish the task",
                max_episodes=max_online_episodes_added_per_iter,
            )

            offline_iterator = offline_replay_buffer.get_iterator(
                batch_size=batch_size, async_prefetch=async_prefetch, queue_size=2
            )

            # Push parameters to actor server (atomic: before buffer reinit)
            # Note: We put this before BC finetune for better stability of the training loop
            push_actor_policy_to_queue(parameters_queue=parameters_queue, policy=policy)
            logging.info("[OFFLINE] Pushed updated policy parameters to actor.")

            # --------- Optional BC finetune on merged buffer ---------
            # if not has_successful_online_episodes:
            #     logging.info("[OFFLINE] No successful online episodes found. Skipping BC finetune.")

            if bc_steps_after_merge > 0:
                logging.info(f"[OFFLINE] Running BC finetune for {bc_steps_after_merge} steps...")
                # Unfreeze encoder for BC stage (full-model BC)
                original_encoder_requires_grad = None
                if hasattr(policy.actor.encoder, "image_encoder") and len(list(policy.actor.encoder.image_encoder.parameters())) > 0:
                    original_encoder_requires_grad = next(policy.actor.encoder.image_encoder.parameters()).requires_grad
                
                # Ensure actor encoder is trainable
                policy._ensure_actor_encoder_trainable()

                for bc_step in tqdm(range(bc_steps_after_merge), desc="BC finetune"):
                    bc_batch = next(offline_iterator)
                    bc_forward_batch = {
                        "action": bc_batch["action"],
                        "state": bc_batch["state"],
                    }
                    observation_features, next_observation_features = get_observation_features(
                        policy=policy, observations=bc_batch["state"], next_observations=bc_batch["next_state"]
                    )
                    bc_forward_batch["observation_feature"] = observation_features
                    bc_forward_batch["next_observation_feature"] = next_observation_features

                    bc_out = policy.forward(bc_forward_batch, model="actor_bc")
                    optimizers["actor"].zero_grad()
                    bc_out["loss_actor_bc"].backward()
                    clip_grad_norm_(policy.actor.parameters(), clip_grad_norm_value)
                    optimizers["actor"].step()

                    if wandb_logger is not None and bc_step % log_freq == 0:
                        wandb_logger.log_dict(
                            {
                                "bc_loss": bc_out["loss_actor_bc"].item(),
                                "bc_step": bc_step,
                                "Optimization step": optimization_step,
                            },
                            mode="train",
                            custom_step_key="Optimization step",
                        )

                    # Process all available transitions to the replay buffer, send by the actor server
                    process_transitions(
                        transition_queue=transition_queue,
                        replay_buffer=replay_buffer,
                        offline_replay_buffer=offline_replay_buffer,
                        device=storage_device,
                        dataset_repo_id=dataset_repo_id,
                        shutdown_event=shutdown_event,
                    )

                    # Process all available interaction messages sent by the actor server
                    interaction_message = process_interaction_messages(
                        interaction_message_queue=interaction_message_queue,
                        interaction_step_shift=interaction_step_shift,
                        wandb_logger=wandb_logger,
                        shutdown_event=shutdown_event,
                    )

                # Reset encoder requires_grad to original value if it was set
                if original_encoder_requires_grad is not None:
                    for param in policy.actor.encoder.image_encoder.parameters():
                        param.requires_grad_(original_encoder_requires_grad)
                
                # Optionally sync encoders if not shared
                if cfg.offline.sync_critic_encoder_after_bc and not policy.shared_encoder:
                    logging.info("[OFFLINE] Syncing critic encoder with actor encoder after BC...")
                    policy.encoder_critic.load_state_dict(policy.actor.encoder.state_dict(), strict=False)

                logging.info("[OFFLINE] BC finetune complete.")

            # Update target networks after BC (critical for value/critic targets)
            policy.update_target_networks()
            logging.info("[OFFLINE] Target networks updated after BC.")

            # Clear online buffer to start fresh for next iteration
            # Note: We clear instead of recreating to maintain the same object reference.
            # The actor connection is through transition_queue, not the buffer object itself,
            # so clearing is safe and more efficient.
            replay_buffer.clear()
            logging.info("[OFFLINE] Cleared online replay buffer.")

            # Log acceptance metrics
            if wandb_logger is not None:
                wandb_logger.log_dict(
                    {
                        "ope_score": cand_score,
                        "ope_frames": cand_frames,
                        "ope_improvement": improvement,
                        "Optimization step": optimization_step,
                    },
                    mode="train",
                    custom_step_key="Optimization step",
                )

        else:
            # Reject candidate; keep collecting more online episodes into online_buffer
            logging.info(
                f"[OFFLINE] Policy rejected (insufficient improvement or frames). "
                f"Current: {cand_score:.2f}, Improvement: {improvement:.3f}, threshold_fraction: {ope_adaptive_threshold_fraction}"
                f"Frames: {cand_frames}"
            )



def start_learner(
    parameters_queue: Queue,
    transition_queue: Queue,
    interaction_message_queue: Queue,
    shutdown_event: any,  # Event,
    cfg: TrainRLServerPipelineConfig,
):
    """
    Start the learner server for training.
    It will receive transitions and interaction messages from the actor server,
    and send policy parameters to the actor server.

    Args:
        parameters_queue: Queue for sending policy parameters to the actor
        transition_queue: Queue for receiving transitions from the actor
        interaction_message_queue: Queue for receiving interaction messages from the actor
        shutdown_event: Event to signal shutdown
        cfg: Training configuration
    """
    if not use_threads(cfg):
        # Create a process-specific log file
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"learner_process_{os.getpid()}.log")

        # Initialize logging with explicit log file
        init_logging(log_file=log_file, display_pid=True)
        logging.info("Learner server process logging initialized")

        # Setup process handlers to handle shutdown signal
        # But use shutdown event from the main process
        # Return back for MP
        # TODO: Check if its useful
        _ = ProcessSignalHandler(False, display_pid=True)

    service = LearnerService(
        shutdown_event=shutdown_event,
        parameters_queue=parameters_queue,
        seconds_between_pushes=cfg.policy.actor_learner_config.policy_parameters_push_frequency,
        transition_queue=transition_queue,
        interaction_message_queue=interaction_message_queue,
        queue_get_timeout=cfg.policy.actor_learner_config.queue_get_timeout,
    )

    server = grpc.server(
        ThreadPoolExecutor(max_workers=MAX_WORKERS),
        options=[
            ("grpc.max_receive_message_length", MAX_MESSAGE_SIZE),
            ("grpc.max_send_message_length", MAX_MESSAGE_SIZE),
        ],
    )

    services_pb2_grpc.add_LearnerServiceServicer_to_server(
        service,
        server,
    )

    host = cfg.policy.actor_learner_config.learner_host
    port = cfg.policy.actor_learner_config.learner_port

    server.add_insecure_port(f"{host}:{port}")
    server.start()
    logging.info("[LEARNER] gRPC server started")

    shutdown_event.wait()
    logging.info("[LEARNER] Stopping gRPC server...")
    server.stop(SHUTDOWN_TIMEOUT)
    logging.info("[LEARNER] gRPC server stopped")


def make_optimizers_and_scheduler(cfg: TrainRLServerPipelineConfig, policy: nn.Module):
    """
    Creates and returns optimizers for the actor, critic, and temperature components of a reinforcement learning policy.

    This function sets up Adam optimizers for:
    - The **actor network**, ensuring that only relevant parameters are optimized.
    - The **critic ensemble**, which evaluates the value function.
    - The **temperature parameter**, which controls the entropy in soft actor-critic (SAC)-like methods.

    It also initializes a learning rate scheduler, though currently, it is set to `None`.

    NOTE:
    - If the encoder is shared, its parameters are excluded from the actor's optimization process.
    - The policy's log temperature (`log_alpha`) is wrapped in a list to ensure proper optimization as a standalone tensor.

    Args:
        cfg: Configuration object containing hyperparameters.
        policy (nn.Module): The policy model containing the actor, critic, and temperature components.

    Returns:
        Tuple[Dict[str, torch.optim.Optimizer], Optional[torch.optim.lr_scheduler._LRScheduler]]:
        A tuple containing:
        - `optimizers`: A dictionary mapping component names ("actor", "critic", "temperature") to their respective Adam optimizers.
        - `lr_scheduler`: Currently set to `None` but can be extended to support learning rate scheduling.

    """
    optimizers = cfg.optimizer.build(policy.get_optim_params())
    scheduler = None if cfg.scheduler is None else cfg.scheduler.build(optimizers["actor"], cfg.steps)

    return optimizers, scheduler




if __name__ == "__main__":
    train_cli()
    logging.info("[LEARNER] main finished")
