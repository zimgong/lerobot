#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team.
# All rights reserved.
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
Offline learner for the HILSerl pipeline.

This script implements the Implicit Q-Learning (IQL) offline stage described in
RL-100 / Uni-O4, using an Advantage Weighted Regression (AWR) actor on top of the
Gaussian policy already employed by the SAC learner. It reuses the same policy
factory and dataset preprocessing utilities, but replaces the online replay buffer
and gRPC actor communication with a static dataset iterator.
"""

import logging
import os
from pathlib import Path

import torch
import torch.optim
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import make_policy
from lerobot.rl.buffer import ReplayBuffer
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.utils.constants import (
    ACTION,
    CHECKPOINTS_DIR,
    LAST_CHECKPOINT_LINK,
    TRAINING_STATE_DIR,
)
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    load_training_state as utils_load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import format_big_number, init_logging


@parser.wrap()
def train_cli(cfg: TrainRLServerPipelineConfig) -> None:
    train(cfg)


def train(cfg: TrainRLServerPipelineConfig) -> None:
    cfg.validate()

    os.makedirs(cfg.output_dir, exist_ok=True)
    log_dir = Path(cfg.output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"offline_{cfg.job_name}.log"
    init_logging(log_file=str(log_file))

    logging.info("[OFFLINE] Configuration:")
    logging.info(cfg.to_dict())

    if cfg.wandb.enable and cfg.wandb.project:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info("[OFFLINE] W&B disabled, logging locally.")

    set_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    dataset = make_dataset(cfg)
    policy = make_policy(cfg.policy, ds_meta=dataset.meta)
    policy.train()

    replay_buffer = ReplayBuffer.from_lerobot_dataset(
        lerobot_dataset=dataset,
        device=cfg.policy.device,
        state_keys=cfg.policy.input_features.keys(),
        capacity=len(dataset),
        storage_device=cfg.policy.storage_device,
        use_drq=False,
        optimize_memory=True,
    )

    logging.info(f"[OFFLINE] Loaded dataset with {len(dataset)} transitions.")
    log_training_info(cfg, policy)

    optimizers = cfg.optimizer.build(policy.get_optim_params())
    scheduler = None if cfg.scheduler is None else cfg.scheduler.build(optimizers["actor"], cfg.steps)

    start_step = 0
    if cfg.resume:
        start_step, optimizers, scheduler = resume_training_state(cfg, optimizers, scheduler)

    iterator = replay_buffer.get_iterator(batch_size=cfg.batch_size, async_prefetch=False)

    grad_clip_norm = cfg.policy.grad_clip_norm
    policy_update_freq = max(1, cfg.policy.policy_update_freq)

    progress = tqdm(range(start_step, cfg.steps), initial=start_step, total=cfg.steps, desc="Offline RL")

    for step in progress:
        batch = next(iterator)

        actions = batch[ACTION]
        observations = batch["state"]
        next_observations = batch["next_state"]
        rewards = batch["reward"]
        done = batch["done"]

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
            "complementary_info": batch.get("complementary_info"),
        }

        value_output = policy.forward(forward_batch, model="value")
        loss_value = value_output["loss_value"]
        optimizers["value"].zero_grad()
        loss_value.backward()
        value_grad_norm = clip_grad_norm_(policy.value_head.parameters(), grad_clip_norm).item()
        optimizers["value"].step()

        critic_output = policy.forward(forward_batch, model="critic")
        loss_critic = critic_output["loss_critic"]
        optimizers["critic"].zero_grad()
        loss_critic.backward()
        critic_grad_norm = clip_grad_norm_(policy.critic_ensemble.parameters(), grad_clip_norm).item()
        optimizers["critic"].step()

        actor_output = policy.forward(forward_batch, model="actor")
        loss_actor = actor_output["loss_actor"]
        actor_grad_norm = None
        if (step + 1) % policy_update_freq == 0:
            optimizers["actor"].zero_grad()
            loss_actor.backward()
            actor_grad_norm = clip_grad_norm_(
                [p for n, p in policy.actor.named_parameters() if not policy.shared_encoder or not n.startswith("encoder")],
                grad_clip_norm,
            ).item()
            optimizers["actor"].step()

        policy.update_target_networks()

        metrics = {
            "loss_value": loss_value.item(),
            "loss_critic": loss_critic.item(),
            "loss_actor": loss_actor.item(),
            "value_grad_norm": value_grad_norm,
            "critic_grad_norm": critic_grad_norm,
        }

        if actor_grad_norm is not None:
            metrics["actor_grad_norm"] = actor_grad_norm

        metrics.update(critic_output.get("q_info", {}))
        metrics.update(actor_output.get("actor_info", {}))
        metrics["buffer_size"] = len(replay_buffer)
        metrics["Training step"] = step + 1

        progress.set_postfix(
            {
                "loss_c": f"{metrics['loss_critic']:.3f}",
                "loss_v": f"{metrics['loss_value']:.3f}",
                "loss_a": f"{metrics['loss_actor']:.3f}",
            }
        )

        if wandb_logger is not None:
            wandb_logger.log_dict(metrics, mode="train", custom_step_key="Training step")

        if scheduler is not None:
            scheduler.step()

        if cfg.save_checkpoint and (step + 1) % cfg.save_freq == 0:
            checkpoint_dir = get_step_checkpoint_dir(Path(cfg.output_dir), cfg.steps, step + 1)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                step=step + 1,
                cfg=cfg,
                policy=policy,
                optimizer=optimizers,
                scheduler=scheduler,
            )
            update_last_checkpoint(checkpoint_dir)

    if cfg.save_checkpoint and cfg.steps % cfg.save_freq != 0:
        checkpoint_dir = get_step_checkpoint_dir(Path(cfg.output_dir), cfg.steps, cfg.steps)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        save_checkpoint(
            checkpoint_dir=checkpoint_dir,
            step=cfg.steps,
            cfg=cfg,
            policy=policy,
            optimizer=optimizers,
            scheduler=scheduler,
        )
        update_last_checkpoint(checkpoint_dir)

    logging.info("[OFFLINE] Training complete.")


def resume_training_state(
    cfg: TrainRLServerPipelineConfig,
    optimizers: dict[str, torch.optim.Optimizer],
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
) -> tuple[int, dict[str, torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler | None]:
    checkpoint_dir = Path(cfg.output_dir) / CHECKPOINTS_DIR / LAST_CHECKPOINT_LINK
    if not checkpoint_dir.exists():
        logging.warning("[OFFLINE] Resume requested but no checkpoint found. Starting from scratch.")
        return 0, optimizers, scheduler

    try:
        step, loaded_optimizers, loaded_scheduler = utils_load_training_state(checkpoint_dir, optimizers, scheduler)
        if isinstance(optimizers, dict):
            optimizers.update(loaded_optimizers)
        else:
            optimizers = loaded_optimizers
        scheduler = loaded_scheduler
        logging.info(f"[OFFLINE] Resuming from checkpoint {checkpoint_dir} at step {step}.")
        return step, optimizers, scheduler
    except Exception as error:  # noqa: BLE001
        logging.error(f"[OFFLINE] Failed to resume from checkpoint: {error}")
        return 0, optimizers, scheduler


def get_observation_features(
    policy,
    observations: dict[str, torch.Tensor],
    next_observations: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor] | None, dict[str, torch.Tensor] | None]:
    if policy.config.vision_encoder_name is None or not policy.config.freeze_vision_encoder:
        return None, None

    with torch.no_grad():
        obs_features = policy.actor.encoder.get_cached_image_features(observations)
        next_obs_features = policy.actor.encoder.get_cached_image_features(next_observations)
    return obs_features, next_obs_features


def log_training_info(cfg: TrainRLServerPipelineConfig, policy: torch.nn.Module) -> None:
    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    logging.info(f"{cfg.policy.device=}")
    logging.info(f"{cfg.steps=}")
    logging.info(f"learnable_params={num_learnable_params} ({format_big_number(num_learnable_params)})")
    logging.info(f"total_params={num_total_params} ({format_big_number(num_total_params)})")


if __name__ == "__main__":
    train_cli()
