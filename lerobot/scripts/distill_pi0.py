#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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
Pi0 One-Step Distillation: Training a direct action prediction model from Pi0 teacher.

This script adapts the LeRobot training pipeline to perform knowledge distillation 
from a pre-trained Pi0 flow matching model to a one-step model.

Example usage:
```bash
python lerobot/scripts/distill_pi0.py \
  --teacher=lerobot/pi0 \
  --policy.type=pi0_onestep \
  --output_dir=lerobot/pi0_onestep \
  --dataset.repo_id=danaaubakirova/koch_test \
  --steps=10000 \
  --batch_size=32 \
  --temperature=1.0 \
  --soft_target_weight=0.5 \
  --hard_target_weight=0.5
```
"""

import logging
import time
import datetime as dt
from pathlib import Path
import os
from contextlib import nullcontext
from dataclasses import dataclass, field
from pprint import pformat
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.amp import GradScaler
from torch.optim import Optimizer
from termcolor import colored

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.utils import cycle
from lerobot.common.envs.factory import make_env
from lerobot.common.optim.factory import make_optimizer_and_scheduler
from lerobot.common.policies.factory import make_policy
# from lerobot.common.policies.factory import register_policy_class
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy, PI0FlowMatching
from lerobot.common.policies.pi0.modeling_onesteppi0 import PI0OneStepConfig, PI0OneStepModel, PI0OneStepPolicy
from lerobot.common.policies.pi0.configuration_pi0 import PI0Config
from lerobot.common.constants import ACTION, OBS_ROBOT
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.common.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    has_method,
    init_logging,
)
from lerobot.common.utils.wandb_utils import WandBLogger
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.eval import eval_policy
from lerobot.configs.policies import PreTrainedConfig
from tqdm import tqdm


@dataclass
class DistillPipelineConfig(TrainPipelineConfig):
    """Configuration for Pi0 distillation pipeline."""
    
    teacher: Optional[str] = None
    teacher_device: str = "cuda"
    freeze_teacher: bool = True
    
    temperature: float = 1.0
    soft_target_weight: float = 0.5
    hard_target_weight: float = 0.5
    # Add epoch-based training configuration
    epochs: int = 10  # Number of epochs to train
    eval_freq_epochs: float = 0.5  # Evaluate every 0.5 epochs



def update_policy_with_distillation(
    train_metrics: MetricsTracker,
    policy: PI0OneStepPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    grad_scaler: GradScaler,
    temperature: float = 1.0,
    soft_target_weight: float = 0.5,
    hard_target_weight: float = 0.5,
    lr_scheduler=None,
    use_amp: bool = False,
    lock=None,
) -> tuple[MetricsTracker, dict]:
    """Modified update_policy function that includes distillation parameters"""
    start_time = time.perf_counter()
    device = get_device_from_parameters(policy)
    policy.train()
    
    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        loss, output_dict = policy.forward(
            batch, 
            temperature=temperature,
            soft_weight=soft_target_weight,
            hard_weight=hard_target_weight
        )
        
    grad_scaler.scale(loss).backward()

    # Unscale the gradient of the optimizer's assigned params in-place
    grad_scaler.unscale_(optimizer)

    grad_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(),
        grad_clip_norm,
        error_if_nonfinite=False,
    )

    with lock if lock is not None else nullcontext():
        grad_scaler.step(optimizer)
    grad_scaler.update()

    optimizer.zero_grad()

    # Step through scheduler if provided
    if lr_scheduler is not None:
        lr_scheduler.step()

    if has_method(policy, "update"):
        policy.update()

    train_metrics.loss = loss.item()
    if 'soft_loss' in output_dict:
        train_metrics.soft_loss = output_dict['soft_loss']
    if 'hard_loss' in output_dict:
        train_metrics.hard_loss = output_dict['hard_loss']
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    
    return train_metrics, output_dict


@parser.wrap()
def distill_pi0(cfg: DistillPipelineConfig):
    """Main distillation training function"""
    # Register our policy class before validation
    # register_policy_class(PI0OneStepPolicy)
    
    cfg.validate()
    logging.info(pformat(cfg.to_dict()))

    if cfg.wandb.enable and cfg.wandb.project:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)
    teacher_device = get_safe_torch_device(cfg.teacher_device, log=True)
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Creating dataset")
    dataset = make_dataset(cfg)

    # Create environment for eval if needed
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None:
        logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size)

    # Load teacher model
    logging.info("Loading teacher model")
    teacher_policy = None
    if cfg.teacher is not None:
        teacher_policy = PI0Policy.from_pretrained(cfg.teacher)
        teacher_policy.to(teacher_device)
        teacher_policy.eval()
        
        # Freeze teacher parameters if specified
        if cfg.freeze_teacher:
            logging.info("Freezing teacher model parameters")
            for param in teacher_policy.parameters():
                param.requires_grad = False
        else:
            logging.info("Teacher model parameters will be updated during training")

    # Create student model
    logging.info("Creating student policy")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
    )
    # Initialize student policy with teacher policy weights if available
    if teacher_policy is not None:
        logging.info("Initializing student policy with teacher policy weights")
        policy.load_state_dict(teacher_policy.state_dict(), strict=False)
    
    # Set teacher policy
    if isinstance(policy, PI0OneStepPolicy):
        policy.teacher_policy = teacher_policy
    else:
        logging.warning(f"Policy type {type(policy)} is not PI0OneStepPolicy. Teacher will not be used.")
    
    policy.to(device)

    logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)

    step = 0  # number of policy updates

    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    if cfg.env is not None:
        logging.info(f"{cfg.env.task=}")
    logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
    logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
    logging.info(f"{dataset.num_episodes=}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")
    
    # Configure data loader
    shuffle = True
    sampler = None

    # Use EpisodeAwareSampler if needed
    if hasattr(cfg.policy, "drop_n_last_frames"):
        from lerobot.common.datasets.sampler import EpisodeAwareSampler
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.episode_data_index,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )

    # Calculate number of iterations per epoch and total steps
    num_iterations_per_epoch = len(dataloader)
    total_steps = num_iterations_per_epoch * cfg.epochs
    
    # Calculate evaluation frequency in terms of steps
    eval_freq_steps = int(num_iterations_per_epoch * cfg.eval_freq_epochs)
    
    # Set save frequency to be at the end of each epoch
    save_freq_steps = num_iterations_per_epoch
    
    # Set log frequency to log 10 times per epoch
    log_freq_steps = max(1, num_iterations_per_epoch // 10)
    
    step = 0  # number of policy updates

    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)
        # Calculate current epoch
        current_epoch = step // num_iterations_per_epoch
        logging.info(f"Resuming from step {step} (epoch {current_epoch})")

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    if cfg.env is not None:
        logging.info(f"{cfg.env.task=}")
    logging.info(f"{cfg.epochs=}")
    logging.info(f"Steps per epoch: {num_iterations_per_epoch}")
    logging.info(f"Total steps: {total_steps}")
    logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
    logging.info(f"{dataset.num_episodes=}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")


    dl_iter = cycle(dataloader)

    policy.train()

    # Define metrics to track
    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "soft_loss": AverageMeter("soft", ":.3f"),
        "hard_loss": AverageMeter("hard", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    train_tracker = MetricsTracker(
        cfg.batch_size, dataset.num_frames, dataset.num_episodes, train_metrics, initial_step=step
    )

    logging.info("Start distillation training")
    for epoch in range(cfg.epochs):
        epoch_start_time = time.time()
        epoch_step = 0
        
        # Create progress bar for this epoch
        progress_bar = tqdm(
            dataloader, 
            desc=f"Epoch {epoch+1}/{cfg.epochs}", 
            leave=True,
            total=num_iterations_per_epoch
        )
        
        # Iterate through the dataloader for this epoch
        for batch in progress_bar:
            epoch_step += 1
            step += 1
            
            start_time = time.perf_counter()
            train_tracker.dataloading_s = time.perf_counter() - start_time

            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device, non_blocking=True)

            # Update policy with this batch
            train_tracker, output_dict = update_policy_with_distillation(
                train_tracker,
                policy,
                batch,
                optimizer,
                cfg.optimizer.grad_clip_norm,
                grad_scaler=grad_scaler,
                temperature=cfg.temperature,
                soft_target_weight=cfg.soft_target_weight,
                hard_target_weight=cfg.hard_target_weight,
                lr_scheduler=lr_scheduler,
                use_amp=cfg.policy.use_amp,
            )

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{output_dict.get('total_loss', 0.0):.4f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
            })
            
            train_tracker.step()
            
            # Determine if this step needs logging, saving, or evaluation
            is_log_step = log_freq_steps > 0 and step % log_freq_steps == 0
            is_saving_step = save_freq_steps > 0 and (step % save_freq_steps == 0 or step == total_steps)
            is_eval_step = eval_freq_steps > 0 and step % eval_freq_steps == 0
            is_end_of_epoch = epoch_step == num_iterations_per_epoch

            if is_log_step:
                # Calculate current epoch progress as a float (e.g., 1.5 = epoch 1, 50% complete)
                current_epoch_progress = epoch + (epoch_step / num_iterations_per_epoch)
                
                logging.info(f"Epoch {current_epoch_progress:.2f}: {train_tracker}")
                if wandb_logger:
                    wandb_log_dict = train_tracker.to_dict()
                    wandb_log_dict["epoch"] = current_epoch_progress
                    if output_dict:
                        wandb_log_dict.update(output_dict)
                    wandb_logger.log_dict(wandb_log_dict, step)
                train_tracker.reset_averages()

            if cfg.save_checkpoint and is_saving_step:
                logging.info(f"Checkpoint at epoch {epoch+1}, step {step}")
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, total_steps, step)
                save_checkpoint(checkpoint_dir, step, cfg, policy, optimizer, lr_scheduler)
                update_last_checkpoint(checkpoint_dir)
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)

            if cfg.env and is_eval_step:
                epoch_progress = epoch + (epoch_step / num_iterations_per_epoch)
                step_id = f"epoch_{epoch_progress:.2f}"
                logging.info(f"Eval at epoch {epoch_progress:.2f}, step {step}")
                with (
                    torch.no_grad(),
                    torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext(),
                ):
                    eval_info = eval_policy(
                        eval_env,
                        policy,
                        cfg.eval.n_episodes,
                        videos_dir=cfg.output_dir / "eval" / f"videos_{step_id}",
                        max_episodes_rendered=4,
                        start_seed=cfg.seed,
                    )

                eval_metrics = {
                    "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                    "pc_success": AverageMeter("success", ":.1f"),
                    "eval_s": AverageMeter("eval_s", ":.3f"),
                }
                eval_tracker = MetricsTracker(
                    cfg.batch_size, dataset.num_frames, dataset.num_episodes, eval_metrics, initial_step=step
                )
                eval_tracker.eval_s = eval_info["aggregated"].pop("eval_s")
                eval_tracker.avg_sum_reward = eval_info["aggregated"].pop("avg_sum_reward")
                eval_tracker.pc_success = eval_info["aggregated"].pop("pc_success")
                logging.info(eval_tracker)
                if wandb_logger:
                    wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                    wandb_log_dict["epoch"] = epoch_progress
                    wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                    wandb_logger.log_video(eval_info["video_paths"][0], step, mode="eval")
        
        # End of epoch
        epoch_time = time.time() - epoch_start_time
        logging.info(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")
        
        # Always save at the end of an epoch
        if cfg.save_checkpoint:
            logging.info(f"Saving checkpoint at end of epoch {epoch+1}")
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, total_steps, step)
            save_checkpoint(checkpoint_dir, step, cfg, policy, optimizer, lr_scheduler)
            update_last_checkpoint(checkpoint_dir)
            if wandb_logger:
                wandb_logger.log_policy(checkpoint_dir)
    if eval_env:
        eval_env.close()
    logging.info("End of distillation training")


if __name__ == "__main__":
    init_logging()
    distill_pi0()