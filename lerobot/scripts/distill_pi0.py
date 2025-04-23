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
from a pre-trained Pi0 flow matching model to a one-step model using DeepSpeed for distributed training.

Example usage:
```bash
deepspeed --num_gpus=4 lerobot/scripts/distill_pi0_deepspeed.py \
  --teacher=lerobot/pi0 \
  --policy.type=pi0_onestep \
  --output_dir=lerobot/pi0_onestep \
  --dataset.repo_id=danaaubakirova/koch_test \
  --epochs=10 \
  --batch_size=32 \
  --temperature=1.0 \
  --soft_target_weight=0.5 \
  --hard_target_weight=0.5 \
  --deepspeed \
  --deepspeed_config=ds_config.json
```
"""

import logging
import time
import datetime as dt
from pathlib import Path
import os
import json
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

# DeepSpeed imports
import deepspeed
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

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
from torch.utils.data.distributed import DistributedSampler


@dataclass
class DistillDeepSpeedPipelineConfig(TrainPipelineConfig):
    """Configuration for Pi0 distillation pipeline with DeepSpeed."""
    
    teacher: Optional[str] = None
    freeze_teacher: bool = True
    
    temperature: float = 1.0
    soft_target_weight: float = 0.5
    hard_target_weight: float = 0.5
    
    # Add epoch-based training configuration
    epochs: int = 10  # Number of epochs to train
    eval_freq_epochs: float = 0.5  # Evaluate every 0.5 epochs
    
    # DeepSpeed configuration
    deepspeed: bool = True
    deepspeed_config: Optional[str] = None
    local_rank: int = -1  # Will be set by DeepSpeed launcher


def create_ds_config(cfg):
    """Create a DeepSpeed configuration if not provided."""
    if cfg.deepspeed_config is not None and os.path.exists(cfg.deepspeed_config):
        return json.load(open(cfg.deepspeed_config, 'r'))
    
    # Get world size for automatic batch size calculation
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    # Default DeepSpeed configuration
    ds_config = {
        "train_batch_size": cfg.batch_size * world_size,
        "gradient_accumulation_steps": 1,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": cfg.optimizer.lr,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": cfg.optimizer.weight_decay
            }
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": cfg.optimizer.lr,
                "warmup_num_steps": cfg.optimizer.num_warmup_steps if hasattr(cfg.optimizer, "num_warmup_steps") else 1000
            }
        },
        "gradient_clipping": cfg.optimizer.grad_clip_norm,
        "fp16": {
            "enabled": True,
            "auto_cast": True,
            "loss_scale": 0,
            "initial_scale_power": 16
        },
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True
        }
    }
    
    return ds_config


@parser.wrap()
def distill_pi0_deepspeed(cfg: DistillDeepSpeedPipelineConfig):
    """Main distillation training function with DeepSpeed"""
    # Initialize DeepSpeed distributed environment
    deepspeed.init_distributed()
    
    # Set local_rank from DeepSpeed
    if "LOCAL_RANK" in os.environ:
        cfg.local_rank = int(os.environ["LOCAL_RANK"])
    
    # Determine if this is the main process
    is_main_process = cfg.local_rank in [0, -1]
    
    # Validate configuration
    cfg.validate()
    if is_main_process:
        logging.info(pformat(cfg.to_dict()))

    # Set up logging and wandb only on main process
    if is_main_process and cfg.wandb.enable and cfg.wandb.project:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        if is_main_process:
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    # Set seed for reproducibility
    if cfg.seed is not None:
        set_seed(cfg.seed + cfg.local_rank if cfg.local_rank != -1 else cfg.seed)

    # Set device based on local_rank
    if cfg.local_rank != -1:
        device = torch.device(f"cuda:{cfg.local_rank}")
        torch.cuda.set_device(device)
    else:
        device = get_safe_torch_device(cfg.policy.device, log=is_main_process)
    
    teacher_device = device
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Create dataset
    if is_main_process:
        logging.info("Creating dataset")
    dataset = make_dataset(cfg)

    # Create environment for evaluation (only on main process)
    eval_env = None
    if is_main_process and cfg.eval_freq_epochs > 0 and cfg.env is not None:
        logging.info("Creating env for evaluation")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size)

    # Load teacher model
    if is_main_process:
        logging.info("Loading teacher model")
    teacher_policy = None
    if cfg.teacher is not None:
        teacher_policy = PI0Policy.from_pretrained(cfg.teacher)
        
        # Put teacher on the same device as the local process
        if cfg.local_rank != -1:
            # In distributed setting, put teacher on local GPU
            teacher_device = device  # Use the local GPU device
        else:
            # In single GPU mode, use the specified teacher device
            teacher_device = get_safe_torch_device(cfg.teacher_device, log=is_main_process)
        
        teacher_policy.to(teacher_device)
        teacher_policy.eval()
        
        # Freeze teacher parameters if specified
        if cfg.freeze_teacher:
            if is_main_process:
                logging.info("Freezing teacher model parameters")
            for param in teacher_policy.parameters():
                param.requires_grad = False
        else:
            if is_main_process:
                logging.info("Teacher model parameters will be updated during training")


    # Create student model
    if is_main_process:
        logging.info("Creating student policy")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
    )
    
    # Initialize student policy with teacher policy weights if available
    if teacher_policy is not None:
        if is_main_process:
            logging.info("Initializing student policy with teacher policy weights")
        policy.load_state_dict(teacher_policy.state_dict(), strict=False)
    
    # Set teacher policy
    if isinstance(policy, PI0OneStepPolicy):
        policy.teacher_policy = teacher_policy
    else:
        if is_main_process:
            logging.warning(f"Policy type {type(policy)} is not PI0OneStepPolicy. Teacher will not be used.")

    # Create or load DeepSpeed config
    ds_config = create_ds_config(cfg) if cfg.deepspeed else None
    
    # Prepare parameters for optimizer
    optimizer_params = [{"params": [p for p in policy.parameters() if p.requires_grad]}]
    
    # Initialize DeepSpeed
    if is_main_process:
        logging.info("Initializing DeepSpeed engine")
    if is_main_process:
        print(ds_config)
    # Initialize DeepSpeed model engine
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=policy,
        model_parameters=optimizer_params,
        config=ds_config if cfg.deepspeed else None,
        optimizer=None,
        lr_scheduler=None
    )
    
    step = 0  # Number of policy updates

    # Resume from checkpoint if needed
    if cfg.resume:
        if is_main_process:
            logging.info(f"Resuming from checkpoint: {cfg.checkpoint_path}")
        
        # Load DeepSpeed checkpoint
        if cfg.deepspeed:
            checkpoint_dir = cfg.checkpoint_path
            if os.path.isfile(checkpoint_dir):
                checkpoint_dir = os.path.dirname(checkpoint_dir)
            
            # Load checkpoint using DeepSpeed's API
            tag = "latest"
            client_state = model_engine.load_checkpoint(
                checkpoint_dir, 
                tag=tag,
                load_optimizer_states=True,
                load_lr_scheduler_states=True
            )
            
            if client_state and "step" in client_state:
                step = client_state["step"]
                if is_main_process:
                    logging.info(f"Resuming from step {step}")
        else:
            # Regular checkpoint loading for non-DeepSpeed mode
            step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    # Log training information
    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    if is_main_process:
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        if cfg.env is not None:
            logging.info(f"{cfg.env.task=}")
        logging.info(f"{cfg.epochs=}")
        logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logging.info(f"{dataset.num_episodes=}")
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")
    
    # Setup distributed sampler or regular sampler
    if cfg.local_rank != -1:
        sampler = DistributedSampler(
            dataset, 
            num_replicas=torch.distributed.get_world_size(),
            rank=torch.distributed.get_rank(),
            shuffle=True
        )
        shuffle = False
    else:
        # For non-distributed training, use regular sampler if needed
        sampler = None
        shuffle = True
        
        # Use EpisodeAwareSampler if needed
        if hasattr(cfg.policy, "drop_n_last_frames"):
            from lerobot.common.datasets.sampler import EpisodeAwareSampler
            shuffle = False
            sampler = EpisodeAwareSampler(
                dataset.episode_data_index,
                drop_n_last_frames=cfg.policy.drop_n_last_frames,
                shuffle=True,
            )

    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=True,
        drop_last=False,
    )

    # Calculate training steps and frequencies
    num_iterations_per_epoch = len(dataloader)
    total_steps = num_iterations_per_epoch * cfg.epochs
    
    # Calculate evaluation frequency in terms of steps
    eval_freq_steps = int(num_iterations_per_epoch * cfg.eval_freq_epochs)
    
    # Set save frequency to be at the end of each epoch
    save_freq_steps = num_iterations_per_epoch
    
    # Set log frequency to log 10 times per epoch
    log_freq_steps = 1

    if is_main_process:
        logging.info(f"Steps per epoch: {num_iterations_per_epoch}")
        logging.info(f"Total steps: {total_steps}")
        logging.info(f"Evaluating every {eval_freq_steps} steps")
        logging.info(f"Saving checkpoints every {save_freq_steps} steps")
        logging.info(f"Logging every {log_freq_steps} steps")

    # Define metrics to track (only on main process)
    if is_main_process:
        train_metrics = {
            "loss": AverageMeter("loss", ":.3f"),
            "soft_loss": AverageMeter("soft", ":.3f"),
            "hard_loss": AverageMeter("hard", ":.3f"),
            "teacher_loss": AverageMeter("hard", ":.3f"),
            "grad_norm": AverageMeter("grdn", ":.3f"),
            "lr": AverageMeter("lr", ":0.1e"),
            "update_s": AverageMeter("updt_s", ":.3f"),
            "dataloading_s": AverageMeter("data_s", ":.3f"),
        }

        train_tracker = MetricsTracker(
            cfg.batch_size, dataset.num_frames, dataset.num_episodes, train_metrics, initial_step=step
        )
    else:
        train_tracker = None

    if is_main_process:
        logging.info("Start distillation training")
    
    # Training loop
    for epoch in range(cfg.epochs):
        epoch_start_time = time.time()
        epoch_step = 0
        
        # Set epoch for distributed sampler
        if sampler is not None and hasattr(sampler, 'set_epoch'):
            sampler.set_epoch(epoch)
        
        # Create progress bar (only on main process)
        if is_main_process:
            progress_bar = tqdm(
                dataloader, 
                desc=f"Epoch {epoch+1}/{cfg.epochs}", 
                leave=True,
                total=num_iterations_per_epoch
            )
        else:
            progress_bar = dataloader
        
        # Iterate through the dataloader for this epoch
        for batch in progress_bar:
            epoch_step += 1
            step += 1
            
            # Time data loading (only on main process)
            if is_main_process:
                start_time = time.perf_counter()
                train_tracker.dataloading_s = time.perf_counter() - start_time
            
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device, non_blocking=True)

            # Training step start time
            if is_main_process:
                update_start_time = time.perf_counter()

            # Forward pass - DeepSpeed will handle the backward and optimizer steps
            loss, output_dict = model_engine(
                batch, 
                temperature=cfg.temperature,
                soft_weight=cfg.soft_target_weight,
                hard_weight=cfg.hard_target_weight
            )
            
            # Backward and optimization (handled by DeepSpeed)
            model_engine.backward(loss)
            model_engine.step()
            
            # Track metrics (only on main process)
            if is_main_process:
                train_tracker.loss = loss.item()
                if 'soft_loss' in output_dict:
                    train_tracker.soft_loss = output_dict['soft_loss']
                if 'hard_loss' in output_dict:
                    train_tracker.hard_loss = output_dict['hard_loss']
                if 'teacher_loss' in output_dict:
                    train_tracker.teacher_loss = output_dict['teacher_loss']
                
                # Get learning rate from optimizer
                train_tracker.lr = model_engine.get_lr()[0] if hasattr(model_engine, 'get_lr') else model_engine.optimizer.param_groups[0]["lr"]
                
                # Calculate update time
                train_tracker.update_s = time.perf_counter() - update_start_time
                
                # Update progress bar
                if isinstance(progress_bar, tqdm):
                    progress_bar.set_postfix({
                        'loss': f"{output_dict.get('total_loss', 0.0):.4f}",
                        'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
                    })
                
                train_tracker.step()
            
            # Determine if this step needs logging, saving, or evaluation
            is_log_step = is_main_process and log_freq_steps > 0 and step % log_freq_steps == 0
            is_saving_step = is_main_process and save_freq_steps > 0 and (step % save_freq_steps == 0 or step == total_steps)
            is_eval_step = is_main_process and eval_freq_steps > 0 and step % eval_freq_steps == 0
            is_end_of_epoch = epoch_step == num_iterations_per_epoch

            # Log metrics
            if is_log_step:
                # Calculate current epoch progress as a float (e.g., 1.5 = epoch 1, 50% complete)
                current_epoch_progress = epoch + (epoch_step / num_iterations_per_epoch)
                
                # logging.info(f"Epoch {current_epoch_progress:.2f}: {train_tracker}")
                if wandb_logger:
                    wandb_log_dict = train_tracker.to_dict()
                    wandb_log_dict["epoch"] = current_epoch_progress
                    if output_dict:
                        for k, v in output_dict.items():
                            if isinstance(v, (int, float)):
                                wandb_log_dict[k] = v
                    wandb_logger.log_dict(wandb_log_dict, step)
                train_tracker.reset_averages()

            # Evaluate model
            if cfg.env and is_eval_step:
                epoch_progress = epoch + (epoch_step / num_iterations_per_epoch)
                step_id = f"epoch_{epoch_progress:.2f}"
                logging.info(f"Eval at epoch {epoch_progress:.2f}, step {step}")
                
                # For evaluation, we need the unwrapped policy module
                eval_policy_module = model_engine.module
                
                with torch.no_grad():
                    eval_info = eval_policy(
                        eval_env,
                        eval_policy_module,
                        cfg.eval.n_episodes,
                        videos_dir=Path(cfg.output_dir) / "eval" / f"videos_{step_id}",
                        max_episodes_rendered=4,
                        start_seed=cfg.seed,
                    )

                # Track evaluation metrics
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
                
                # Log evaluation metrics to WandB
                if wandb_logger:
                    wandb_log_dict = {**eval_tracker.to_dict(), **eval_info["aggregated"]}
                    wandb_log_dict["epoch"] = epoch_progress
                    wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                    if eval_info["video_paths"] and len(eval_info["video_paths"]) > 0:
                        wandb_logger.log_video(eval_info["video_paths"][0], step, mode="eval")
        
        # End of epoch reporting
        if is_main_process:
            epoch_time = time.time() - epoch_start_time
            logging.info(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")
        
        # Always save at the end of an epoch
        if is_main_process and cfg.save_checkpoint:
            logging.info(f"Saving checkpoint at end of epoch {epoch+1}")
            
            # Create checkpoint directory
            checkpoint_dir = Path(cfg.output_dir) / "checkpoints" / f"epoch_{epoch+1}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Save model using DeepSpeed's API
            client_state = {"step": step}
            logging.info("save start")
            try:
                logging.info("Saving model weights directly instead of DeepSpeed checkpoint")
                
                # Get model state dict
                model_state_dict = model_engine.module.state_dict()
                
                # Save weights
                weights_path = checkpoint_dir / "pytorch_model.bin"
                torch.save(model_state_dict, weights_path)
                
                logging.info(f"Successfully saved weights to {weights_path}")
            except Exception as e:
                logging.error(f"Error during weights saving on process {cfg.local_rank}: {str(e)}")
            logging.info("save end")
            
                
            # For Zero-3, save a consolidated checkpoint for easier loading
            if hasattr(model_engine, 'zero_optimization_stage') and model_engine.zero_optimization_stage() > 0:
                # Consolidated weights for non-DeepSpeed use
                fp32_checkpoint_path = checkpoint_dir / "pytorch_model.bin"
                state_dict = model_engine.module.state_dict()
                torch.save(state_dict, fp32_checkpoint_path)
            
            # Save config if available
            if hasattr(policy, "config"):
                try:
                    policy.config.save_pretrained(checkpoint_dir)
                except Exception as e:
                    logging.error(f"Error saving config on process {cfg.local_rank}: {e}")
            
            
            # Log to WandB
            # if wandb_logger:
            #     wandb_logger.log_policy(str(checkpoint_dir))

    # Clean up
    if is_main_process and eval_env:
        eval_env.close()
    
    if is_main_process:
        logging.info("End of distillation training")


if __name__ == "__main__":
    init_logging()
    distill_pi0_deepspeed()