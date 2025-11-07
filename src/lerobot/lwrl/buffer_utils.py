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
Buffer utilities for merging offline and online replay buffers.

This module provides functions to merge successful online episodes into the offline
dataset buffer for iterative offline RL training.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import torch

from lerobot.datasets.dataset_tools import merge_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.rl.buffer import ReplayBuffer
from lerobot.lwrl.buffer_batched import ParallelReplayBuffer


@torch.no_grad()
def merge_offline_online_success(
    offline_buffer: ReplayBuffer | ParallelReplayBuffer,
    online_buffer: ReplayBuffer | ParallelReplayBuffer,
) -> ReplayBuffer | ParallelReplayBuffer:
    """Merge offline buffer with successful online episodes.

    Creates a new replay buffer containing all offline transitions plus
    successful transitions from the online buffer. Both buffers are filtered
    to only include successful episodes before merging.

    Args:
        offline_buffer: Existing offline replay buffer (must be ParallelReplayBuffer)
        online_buffer: Online replay buffer with collected episodes (must be ParallelReplayBuffer)

    Returns:
        New merged replay buffer containing successful episodes from both buffers
    """
    if not isinstance(offline_buffer, ParallelReplayBuffer):
        raise NotImplementedError("Merging offline and online buffers is not supported for ReplayBuffer.")
    
    if not isinstance(online_buffer, ParallelReplayBuffer):
        raise NotImplementedError("Merging offline and online buffers is not supported for ReplayBuffer.")
    
    # Check if buffers are empty
    if len(offline_buffer) == 0 and len(online_buffer) == 0:
        raise ValueError("Both buffers are empty. Cannot merge.")
    
    # Collect datasets to merge (handle empty buffers)
    datasets_to_merge = []
    
    logging.info("Converting buffers to LeRobotDataset (filtering for successful episodes)...")
    # Note: to_lerobot_dataset already filters for successful episodes
    with tempfile.TemporaryDirectory() as tmpdir:
        # Convert offline buffer if not empty
        if len(offline_buffer) > 0:
            offline_dataset_path = Path(tmpdir) / "offline_dataset"
            offline_dataset_path.mkdir(parents=True, exist_ok=True)
            
            offline_dataset = offline_buffer.to_lerobot_dataset(
                repo_id="offline_buffer",
                fps=1,
                root=str(offline_dataset_path),
                task_name="offline",
            )
            
            logging.info(f"Offline dataset: {offline_dataset.meta.total_episodes} episodes, {offline_dataset.meta.total_frames} frames")
            datasets_to_merge.append(offline_dataset)
        
        # Convert online buffer if not empty
        if len(online_buffer) > 0:
            online_dataset_path = Path(tmpdir) / "online_dataset"
            online_dataset_path.mkdir(parents=True, exist_ok=True)
            
            online_dataset = online_buffer.to_lerobot_dataset(
                repo_id="online_buffer",
                fps=1,
                root=str(online_dataset_path),
                task_name="online",
            )
            
            logging.info(f"Online dataset: {online_dataset.meta.total_episodes} episodes, {online_dataset.meta.total_frames} frames")
            datasets_to_merge.append(online_dataset)
        
        # If only one buffer had data, use it directly
        if len(datasets_to_merge) == 1:
            merged_dataset = datasets_to_merge[0]
            logging.info(f"Only one buffer had successful episodes. Using that dataset directly.")
        else:
            # Merge the datasets
            logging.info("Merging datasets...")
            merged_dataset_path = Path(tmpdir) / "merged_dataset"
            merged_dataset_path.mkdir(parents=True, exist_ok=True)
            
            merged_dataset = merge_datasets(
                datasets=datasets_to_merge,
                output_repo_id="merged_buffer",
                output_dir=str(merged_dataset_path),
            )
        
        logging.info(f"Merged dataset: {merged_dataset.meta.total_episodes} episodes, {merged_dataset.meta.total_frames} frames")
        
        # Create new ParallelReplayBuffer from merged dataset
        # Preserve parameters from the original buffers (prefer offline_buffer for consistency)
        logging.info("Creating new ParallelReplayBuffer from merged dataset...")
        merged_buffer = ParallelReplayBuffer.from_lerobot_dataset(
            lerobot_dataset=merged_dataset,
            num_envs=offline_buffer.num_envs,
            device=offline_buffer.device,
            state_keys=offline_buffer.state_keys,
            capacity=None,  # Let it auto-calculate based on dataset size
            image_augmentation_function=offline_buffer.image_augmentation_function,
            use_drq=offline_buffer.use_drq,
            storage_device=offline_buffer.storage_device,
            optimize_memory=offline_buffer.optimize_memory,
        )
        
        # Preserve gamma and n_steps from the original buffer
        merged_buffer.gamma = offline_buffer.gamma
        merged_buffer.n_steps = offline_buffer.n_steps
        
        logging.info(f"Created merged buffer with {len(merged_buffer)} transitions")
        
        return merged_buffer

    
    