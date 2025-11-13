import logging
import os
import time
from pprint import pformat

import torch
from termcolor import colored
from torch import nn
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.policies.factory import make_policy
from lerobot.policies.offline.modeling_offline import OfflineIQLPolicy as CurrentPolicy
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.utils.constants import ACTION
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import get_safe_torch_device, init_logging

from lerobot.lwrl.learner import (
    check_nan_in_transition,
    get_observation_features,
    handle_resume_logic,
    initialize_offline_replay_buffer,
    load_training_state,
    log_training_info,
    save_training_checkpoint,
)


@parser.wrap()
def train_cli(cfg: TrainRLServerPipelineConfig):
    train(cfg, job_name=cfg.job_name)
    logging.info("[OFFLINE_WARMUP] train_cli finished")


def train(cfg: TrainRLServerPipelineConfig, job_name: str | None = None):
    cfg.validate()

    if job_name is None:
        job_name = cfg.job_name
    if job_name is None:
        raise ValueError("Job name must be specified either in config or as a parameter")

    log_dir = os.path.join(cfg.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"warmup_{job_name}.log")
    init_logging(log_file=log_file, display_pid=False)

    logging.info("Offline warmup logging initialized")
    logging.info(pformat(cfg.to_dict()))

    if cfg.wandb.enable and cfg.wandb.project:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    cfg = handle_resume_logic(cfg)

    set_seed(seed=cfg.seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    device = get_safe_torch_device(try_device=cfg.policy.device, log=True)
    storage_device = get_safe_torch_device(try_device=cfg.policy.storage_device)

    clip_grad_norm_value = cfg.policy.grad_clip_norm
    log_freq = cfg.log_freq
    save_freq = cfg.save_freq
    async_prefetch = getattr(cfg.policy, "async_prefetch", False)

    offline_iters = cfg.offline.iters
    iql_steps_per_iter = cfg.offline.iql_steps
    bc_warmup_steps = cfg.offline.bc_steps_after_merge
    saving_checkpoint = cfg.save_checkpoint

    fps = cfg.env.fps if cfg.env is not None else 30

    logging.info("Initializing offline policy for warmup")
    policy: CurrentPolicy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
    )
    assert isinstance(policy, nn.Module)
    policy.train()

    optimizers, lr_scheduler = make_optimizers_and_scheduler(cfg=cfg, policy=policy)
    resume_optimization_step, _ = load_training_state(cfg=cfg, optimizers=optimizers)

    log_training_info(cfg=cfg, policy=policy)

    if cfg.dataset is None:
        raise ValueError("Dataset is required for offline warmup")

    offline_replay_buffer = initialize_offline_replay_buffer(
        cfg=cfg,
        device=device,
        storage_device=storage_device,
    )
    offline_iterator = offline_replay_buffer.get_iterator(
        batch_size=cfg.batch_size, async_prefetch=async_prefetch, queue_size=2
    )

    optimization_step = resume_optimization_step if resume_optimization_step is not None else 0

    dataset_repo_id = cfg.dataset.repo_id

    logging.info("Starting offline warmup loop")

    if bc_warmup_steps > 0:
        logging.info(f"[OFFLINE_WARMUP] Running BC warmup for {bc_warmup_steps} steps")
        if hasattr(policy, "_ensure_actor_encoder_trainable"):
            policy._ensure_actor_encoder_trainable()

        for bc_step in tqdm(range(bc_warmup_steps), desc="BC warmup"):
            batch = next(offline_iterator)

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

    total_iql_steps = iql_steps_per_iter * offline_iters
    progress_bar = tqdm(total=total_iql_steps, desc="IQL warmup")

    for offline_iter in range(offline_iters):
        logging.info(f"[OFFLINE_WARMUP] Iteration {offline_iter + 1}/{offline_iters}")
        for iter_step in range(iql_steps_per_iter):
            time_for_one_optimization_step = time.time()

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

            forward_batch = {
                ACTION: actions,
                "reward": rewards,
                "state": observations,
                "next_state": next_observations,
                "done": done,
                "observation_feature": observation_features,
                "next_observation_feature": next_observation_features,
            }

            training_infos: dict[str, float] = {}

            value_output = policy.forward(forward_batch, model="value")
            loss_value = value_output["loss_value"]
            optimizers["value"].zero_grad()
            loss_value.backward()
            value_grad_norm = clip_grad_norm_(policy.value_head.parameters(), clip_grad_norm_value).item()
            optimizers["value"].step()

            critic_output = policy.forward(forward_batch, model="critic")
            loss_critic = critic_output["loss_critic"]
            optimizers["critic"].zero_grad()
            loss_critic.backward()
            critic_grad_norm = clip_grad_norm_(policy.critic_ensemble.parameters(), clip_grad_norm_value).item()
            optimizers["critic"].step()

            if policy.config.num_discrete_actions is not None:
                discrete_output = policy.forward(forward_batch, model="discrete_critic")
                loss_discrete = discrete_output["loss_discrete_critic"]
                optimizers["discrete_critic"].zero_grad()
                loss_discrete.backward()
                discrete_grad_norm = clip_grad_norm_(
                    policy.discrete_critic.parameters(), clip_grad_norm_value
                ).item()
                optimizers["discrete_critic"].step()
                training_infos["loss_discrete_critic"] = loss_discrete.item()
                training_infos["discrete_critic_grad_norm"] = discrete_grad_norm
                training_infos.update(discrete_output.get("q_info", {}))

            policy.update_target_networks()

            training_infos.update(
                {
                    "loss_value": loss_value.item(),
                    "loss_critic": loss_critic.item(),
                    "value_grad_norm": value_grad_norm,
                    "critic_grad_norm": critic_grad_norm,
                    "offline_replay_buffer_size": len(offline_replay_buffer),
                    "Training step": optimization_step + 1,
                }
            )
            training_infos.update(critic_output.get("q_info", {}))

            if lr_scheduler is not None:
                lr_scheduler.step()

            if optimization_step % log_freq == 0:
                training_infos["Optimization step"] = optimization_step
                if wandb_logger is not None:
                    wandb_logger.log_dict(
                        training_infos,
                        mode="train",
                        custom_step_key="Optimization step",
                    )

            time_for_one_optimization_step = time.time() - time_for_one_optimization_step
            frequency = 1 / (time_for_one_optimization_step + 1e-9)
            if wandb_logger is not None:
                wandb_logger.log_dict(
                    {
                        "Optimization frequency loop [Hz]": frequency,
                        "Optimization step": optimization_step,
                    },
                    mode="train",
                    custom_step_key="Optimization step",
                )

            optimization_step += 1
            progress_bar.update(1)
            
            # Update progress bar with losses (matching offline_learner format)
            postfix_dict = {
                "loss_v": f"{training_infos['loss_value']:.3f}",
                "loss_c": f"{training_infos['loss_critic']:.3f}",
            }
            # Add discrete critic loss if available
            if "loss_discrete_critic" in training_infos:
                postfix_dict["loss_dc"] = f"{training_infos['loss_discrete_critic']:.3f}"
            progress_bar.set_postfix(postfix_dict)

            if saving_checkpoint and (
                optimization_step % save_freq == 0 or optimization_step == total_iql_steps
            ):
                save_training_checkpoint(
                    cfg=cfg,
                    optimization_step=optimization_step,
                    online_steps=total_iql_steps,
                    interaction_message=None,
                    policy=policy,
                    optimizers=optimizers,
                    replay_buffer=offline_replay_buffer,
                    offline_replay_buffer=offline_replay_buffer,
                    dataset_repo_id=dataset_repo_id,
                    fps=fps,
                )

    progress_bar.close()
    logging.info("[OFFLINE_WARMUP] Warmup finished")


def make_optimizers_and_scheduler(cfg: TrainRLServerPipelineConfig, policy: nn.Module):
    optimizers = cfg.optimizer.build(policy.get_optim_params())
    scheduler = None if cfg.scheduler is None else cfg.scheduler.build(optimizers["actor"], cfg.steps)
    return optimizers, scheduler


if __name__ == "__main__":
    train_cli()
    logging.info("[OFFLINE_WARMUP] main finished")
