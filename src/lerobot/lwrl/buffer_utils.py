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
import shutil
import tempfile
from pathlib import Path

import torch

from lerobot.datasets.dataset_tools import merge_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.rl.buffer import ReplayBuffer
from lerobot.lwrl.buffer_batched import ParallelReplayBuffer


@torch.no_grad()
def convert_buffer_to_dataset_with_features(
    buffer: ParallelReplayBuffer,
    repo_id: str,
    root: str,
    task_name: str,
    allowed_features: dict,
    max_episodes: int = -1,
) -> LeRobotDataset:
    """Convert a buffer to LeRobotDataset ensuring it follows the given features.
    
    Validates that the buffer has all required features from allowed_features,
    and filters the dataset to only include those features. Any extra features
    in the buffer will be discarded.
    
    Args:
        buffer: ParallelReplayBuffer to convert
        repo_id: Repository ID for the dataset
        root: Root directory path for the dataset
        task_name: Task name for the dataset
        allowed_features: Dictionary of allowed features. Buffer must have all
            these features, and will be filtered to only include them.
    
    Returns:
        LeRobotDataset with filtered features matching allowed_features
    
    Raises:
        ValueError: If buffer is missing any required features from allowed_features
    """
    # Validate buffer has all required features
    # Get buffer's features by converting temporarily to check compatibility
    temp_path = Path(root).parent / f"{repo_id}_temp_check"
    temp_dataset = buffer.to_lerobot_dataset(
        repo_id=f"{repo_id}_temp",
        fps=1,
        root=str(temp_path),
        task_name=task_name,
        max_episodes=max_episodes,
    )
    buffer_features = temp_dataset.meta.info["features"]
    
    # Check that all allowed features exist in buffer
    missing_features = set(allowed_features.keys()) - set(buffer_features.keys())
    if missing_features:
        raise ValueError(
            f"Buffer is missing required features: {missing_features}. "
            f"Buffer features: {list(buffer_features.keys())}. "
            f"Required features: {list(allowed_features.keys())}"
        )
    
    # Log extra features that will be discarded
    extra_features = set(buffer_features.keys()) - set(allowed_features.keys())
    if extra_features:
        logging.info(f"Buffer has extra features that will be discarded: {extra_features}")
    
    # Clean up temp dataset
    if temp_path.exists():
        shutil.rmtree(temp_path, ignore_errors=True)
    
    # Convert buffer with feature filtering
    dataset = buffer.to_lerobot_dataset(
        repo_id=repo_id,
        fps=1,
        root=root,
        task_name=task_name,
        allowed_features=allowed_features,
    )
    
    return dataset


@torch.no_grad()
def merge_offline_online_success(
    offline_buffer: ReplayBuffer | ParallelReplayBuffer,
    online_buffer: ReplayBuffer | ParallelReplayBuffer,
    allowed_features: dict,
    task_name:str = "Control robot to finish the task",
    max_episodes: int = -1,
) -> ReplayBuffer | ParallelReplayBuffer:
    """Merge offline buffer with successful online episodes.

    Creates a new replay buffer containing all offline transitions plus
    successful transitions from the online buffer. Both buffers are filtered
    to only include successful episodes before merging. Offline buffer must have successful episodes.

    Task name is used to create the merged dataset.
    
    The allowed_features are used as the reference. Online buffer must have
    all required features, and any extra features in online buffer will be discarded.

    Args:
        offline_buffer: Existing offline replay buffer (must be ParallelReplayBuffer)
        online_buffer: Online replay buffer with collected episodes (must be ParallelReplayBuffer)
        allowed_features: Dictionary of allowed features from offline dataset. Online buffer
            will be filtered to only include these features.
        task_name: Task name for the merged dataset.
    Returns:
        New merged replay buffer containing successful episodes from both buffers
    """
    if not isinstance(offline_buffer, ParallelReplayBuffer) or not isinstance(online_buffer, ParallelReplayBuffer):
        raise NotImplementedError("Merging offline and online buffers is not supported for ReplayBuffer.")
    
    # Check successful episodes count before converting to avoid unnecessary work
    offline_success_count = offline_buffer._success_episode_num() if len(offline_buffer) > 0 else 0
    online_success_count = online_buffer._success_episode_num() if len(online_buffer) > 0 else 0

    assert offline_success_count > 0, "Offline buffer must have successful episodes."
    
    if offline_success_count > 0 and online_success_count == 0:
        logging.info("No successful online episodes. Using offline buffer directly.")
        return offline_buffer
    
    logging.info(f"Successful episodes - Offline: {offline_success_count}, Online: {online_success_count}")
    logging.info(f"Using allowed features: {list(allowed_features.keys())}")
    
    # Collect datasets to merge (handle empty buffers)
    datasets_to_merge = []
    
    logging.info("Converting buffers to LeRobotDataset (filtering for successful episodes)...")
    # Note: to_lerobot_dataset() filters for successful episodes by checking
    # complementary_info.is_success == 1.0 in at least one frame per episode.
    # Only episodes with at least one successful frame are included in the dataset.
    with tempfile.TemporaryDirectory() as tmpdir:
        # Convert online buffer if it has successful episodes
        if online_success_count > 0:
            online_dataset_path = Path(tmpdir) / "online_dataset"
            online_dataset = convert_buffer_to_dataset_with_features(
                buffer=online_buffer,
                repo_id="online_buffer",
                root=str(online_dataset_path),
                task_name=task_name,
                allowed_features=allowed_features,
                max_episodes=max_episodes,
            )
            logging.info(f"Online dataset: {online_dataset.meta.total_episodes} episodes, {online_dataset.meta.total_frames} frames")
            datasets_to_merge.append(online_dataset)
        else:
            logging.info("Skipping online buffer conversion (no successful episodes)")
        
        # Convert offline buffer if it has successful episodes
        if offline_success_count > 0:
            offline_dataset_path = Path(tmpdir) / "offline_dataset"
            offline_dataset = convert_buffer_to_dataset_with_features(
                buffer=offline_buffer,
                repo_id="offline_buffer",
                root=str(offline_dataset_path),
                task_name=task_name,
                allowed_features=allowed_features,
            )
            logging.info(f"Offline dataset: {offline_dataset.meta.total_episodes} episodes, {offline_dataset.meta.total_frames} frames")
            datasets_to_merge.append(offline_dataset)
        else:
            logging.info("Skipping offline buffer conversion (no successful episodes)")        

        # Handle merging based on available datasets
        # Merge the datasets
        logging.info("Merging datasets...")
        merged_dataset_path = Path(tmpdir) / "merged_dataset"
        # Don't create directory - merge_datasets() will create it internally
        
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

    
    