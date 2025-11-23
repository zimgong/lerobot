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
Offline Policy Evaluation (OPE) utilities for iterative offline RL.

This module implements AM-Q (Average Model Q-value) scoring for policy evaluation,
which sums Q-values along policy-collected episodes to rank policies offline.
"""

from __future__ import annotations

import torch
from torch import Tensor

from lerobot import envs
from lerobot.rl.buffer import ReplayBuffer
from lerobot.lwrl.buffer_batched import ParallelReplayBuffer
import tqdm

# for online eval
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.utils.utils import get_safe_torch_device
from lerobot.lwrl.sim.lwlab.env_lwlab import make_lwlab_robot_env, make_lwlab_processors
from lerobot.processor.converters import create_transition, TransitionKey
from lerobot.lwrl.sim.lwlab.env_lwlab import step_lwlab_env_and_process_transition

@torch.no_grad()
def amq_score_from_buffer(buf: ReplayBuffer | ParallelReplayBuffer, key: str = "q_min") -> tuple[float, int]:
    """Compute AM-Q score from buffer's complementary_info.

    AM-Q (Average Model Q-value) is computed as the sum of Q-values across
    all transitions, normalized by the number of frames (transitions).
    
    Only sums over valid (initialized) entries to avoid NaN from uninitialized
    buffer slots created with torch.empty().

    Args:
        buf: ReplayBuffer or ParallelReplayBuffer containing transitions with complementary_info
        key: Key in complementary_info containing Q-values (default: "q_min")

    Returns:
        Tuple of (AM-Q score, number of frames). Returns (-inf, 0) if no data.
    """
    if not hasattr(buf, "initialized") or not buf.initialized:
        return float("-inf"), 0

    if not hasattr(buf, "has_complementary_info") or not buf.has_complementary_info:
        return float("-inf"), 0

    qvals: Tensor | None = buf.complementary_info.get(key, None)
    if qvals is None or qvals.numel() == 0:
        return float("-inf"), 0

    # Get total number of frames (transitions) in the buffer
    num_frames = len(buf)
    if num_frames == 0:
        return float("-inf"), 0

    # For ParallelReplayBuffer, we need to mask out uninitialized entries
    # to avoid NaN from torch.empty() initialization
    if isinstance(buf, ParallelReplayBuffer):
        # Sum only over valid entries for each environment
        total_q_sum = 0.0
        for env_idx in range(buf.num_envs):
            env_size = buf.size[env_idx].item()
            if env_size == 0:
                continue
            
            # Get valid indices for this environment (ring buffer)
            pos = buf.position[env_idx].item()
            # Valid indices are from (pos - env_size) % capacity to pos - 1
            start_idx = (pos - env_size) % buf.capacity
            
            # Extract valid qvals for this environment
            if start_idx + env_size <= buf.capacity:
                # No wrap-around: contiguous slice
                env_qvals = qvals[env_idx, start_idx:start_idx + env_size]
            else:
                # Wrap-around: need to concatenate two slices
                slice1 = qvals[env_idx, start_idx:]
                slice2 = qvals[env_idx, :pos]
                env_qvals = torch.cat([slice1, slice2], dim=0)
            
            # Filter out NaN values (safety check)
            valid_mask = torch.isfinite(env_qvals)
            if valid_mask.any():
                total_q_sum += env_qvals[valid_mask].sum().item()
        score = total_q_sum / num_frames if num_frames > 0 else float("-inf")
    else:
        # For regular ReplayBuffer, assume all entries up to len(buf) are valid
        # Still filter NaN for safety
        valid_qvals = qvals[:num_frames]
        valid_mask = torch.isfinite(valid_qvals)
        if valid_mask.any():
            score = float(valid_qvals[valid_mask].sum().item()) / num_frames
        else:
            score = float("-inf")
    
    return score, num_frames


@torch.no_grad()
def amq_score_from_buffer_sample(
    buf: ReplayBuffer | ParallelReplayBuffer,
    policy,  # OfflineIQLPolicy - using Any to avoid circular imports
    num_samples: int,
    adaptive_threshold_fraction: float = 0.05,
) -> tuple[float, int]:
    """Compute AM-Q score by sampling from buffer and evaluating with current policy.
    
    This function samples transitions from the buffer and computes Q-values using
    the current policy's critic ensemble. This is useful for online evaluation where
    we want to evaluate the current policy on collected data.
    
    Args:
        buf: ReplayBuffer or ParallelReplayBuffer containing transitions
        policy: OfflineIQLPolicy to use for computing Q-values
        num_samples: Number of samples to draw from the buffer for evaluation
        
    Returns:
        Tuple of (AM-Q score, number of samples evaluated). Returns (-inf, 0) if no data.
    """
    if not hasattr(buf, "initialized") or not buf.initialized:
        return float("-inf"), 0, 0.0, False
    
    # Get total number of frames in the buffer
    num_frames = len(buf)
    if num_frames == 0:
        return float("-inf"), 0, 0.0, False
    
    # Sample batches from the buffer and accumulate q_min values
    total_q_sum_prev = 0.0
    total_q_sum_current = 0.0
    total_samples = 0
    
    # Sample in batches to be efficient
    batch_size = min(256, num_samples)  # Reasonable batch size
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    for _ in tqdm.tqdm(range(num_batches), desc="Sampling from buffer for OPE"):
        # Sample a batch from the buffer
        batch = buf.sample(batch_size=min(batch_size, num_samples - total_samples))
        
        observations = batch["state"]
        actions = batch["action"]
        
        # Get observation features if needed (for frozen encoders)
        obs_feats = None
        if getattr(policy.config, "freeze_vision_encoder", False) and policy.actor.encoder.has_images:
            obs_feats = policy.actor.encoder.get_cached_image_features(observations)
        
        # Compute Q-values using policy's critic ensemble
        # Q (ensemble), take conservative Q_min
        q_values = policy.critic_ensemble(observations, actions, obs_feats)  # [n_heads, B]
        q_min_prev = q_values.min(dim=0)[0]  # [B]

        # current Q-values: select actions using current policy
        # select_action expects observation keys directly, not nested in "state"
        action = policy.select_action(observations)
        q_values_current = policy.critic_ensemble(observations, action, obs_feats)
        q_min_current = q_values_current.min(dim=0)[0]
        
        # Filter out NaN/Inf values for safety
        valid_mask = torch.isfinite(q_min_prev)
        if valid_mask.any():
            total_q_sum_prev += q_min_prev[valid_mask].sum().item()
            total_q_sum_current += q_min_current[valid_mask].sum().item()
            total_samples += valid_mask.sum().item()
    
    # Compute AM-Q score as mean of q_min values
    if total_samples > 0:
        score_prev = total_q_sum_prev / total_samples
        score_current = total_q_sum_current / total_samples
    else:
        score_prev = float("-inf")
        score_current = float("-inf")

    threshold = score_prev * (1 + adaptive_threshold_fraction)
    improvement = (score_current - score_prev) / score_prev
    if score_current > threshold:
        return score_current, total_samples, improvement, True
    else:
        return score_prev, total_samples, improvement, False


@torch.no_grad()
def amq_score_from_buffer_online(
    buf: ReplayBuffer | ParallelReplayBuffer,
    policy,  # OfflineIQLPolicy - using Any to avoid circular imports
    cfg: TrainRLServerPipelineConfig,
    num_samples: int,
    adaptive_threshold_fraction: float = 0.05,
) -> tuple[float, int]:
    """Compute AM-Q score by sampling from buffer and evaluating with current policy.
    
    This function samples transitions from the buffer and computes Q-values using
    the current policy's critic ensemble. This is useful for online evaluation where
    we want to evaluate the current policy on collected data.
    
    Args:
        buf: ReplayBuffer or ParallelReplayBuffer containing transitions
        policy: OfflineIQLPolicy to use for computing Q-values
        num_samples: Number of samples to draw from the buffer for evaluation
        
    Returns:
        Tuple of (AM-Q score, number of samples evaluated). Returns (-inf, 0) if no data.
    """
    if not hasattr(buf, "initialized") or not buf.initialized:
        return float("-inf"), 0, 0.0, False
    
    # Get total number of frames in the buffer
    num_frames = len(buf)
    if num_frames == 0:
        return float("-inf"), 0, 0.0, False
    
    # Sample batches from the buffer and accumulate q_min values
    total_q_sum_prev = 0.0
    total_q_sum_current = 0.0
    total_samples = 0
    
    # Sample in batches to be efficient
    batch_size = min(cfg.batch_size, num_samples)  # Reasonable batch size
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    for _ in tqdm.tqdm(range(num_batches), desc="Sampling from buffer for OPE"):
        # Sample a batch from the buffer
        batch = buf.sample(batch_size=min(batch_size, num_samples - total_samples))
        
        observations = batch["state"]
        actions = batch["action"]
        
        # Get observation features if needed (for frozen encoders)
        obs_feats = None
        if getattr(policy.config, "freeze_vision_encoder", False) and policy.actor.encoder.has_images:
            obs_feats = policy.actor.encoder.get_cached_image_features(observations)
        
        # Compute Q-values using policy's critic ensemble
        # Q (ensemble), take conservative Q_min
        q_values = policy.critic_ensemble(observations, actions, obs_feats)  # [n_heads, B]
        q_min_prev = q_values.min(dim=0)[0]  # [B]

        # # current Q-values: select actions using current policy
        # # select_action expects observation keys directly, not nested in "state"
        # action = policy.select_action(observations)
        # q_values_current = policy.critic_ensemble(observations, action, obs_feats)
        # q_min_current = q_values_current.min(dim=0)[0]
        
        # Filter out NaN/Inf values for safety
        valid_mask = torch.isfinite(q_min_prev)
        if valid_mask.any():
            total_q_sum_prev += q_min_prev[valid_mask].sum().item()
            # total_q_sum_current += q_min_current[valid_mask].sum().item()
            total_samples += valid_mask.sum().item()

    
    # evaluate the current policy based on another simulator
    # build env based on env_cfg
    env_cfg = cfg.ope_eval_env
    assert env_cfg.type == "lwlab", "LwLab environment must be provided"
    device = get_safe_torch_device(cfg.learner_device, log=True)
    
    online_env, teleop_device = make_lwlab_robot_env(cfg=env_cfg)
    env_processor, action_processor = make_lwlab_processors(online_env, teleop_device, env_cfg, device)
    obs, info = online_env.reset()
    env_processor.reset()
    action_processor.reset()

    # Process initial observation
    transition = create_transition(
        observation=obs,
        reward=torch.zeros((online_env.num_envs,), dtype=torch.float32, device=device),
        done=torch.zeros((online_env.num_envs,), dtype=torch.bool, device=device),
        truncated=torch.zeros((online_env.num_envs,), dtype=torch.bool, device=device),
        info=info)
    transition = env_processor(transition)

    batch_size = min(env_cfg.num_envs, num_samples)
    batch_num = (num_samples + batch_size - 1) // batch_size


    for i in tqdm.tqdm(range(batch_num), desc="Calculating Q-values for OPE on online env"):
        observation = {
            k: v.to(device) for k, v in transition[TransitionKey.OBSERVATION].items() if k in cfg.policy.input_features
        }

        # Get observation features if needed (for frozen encoders)
        # cached features if the encoder is frozen
        with torch.no_grad():
            obs_feats = None
            if getattr(policy.config, "freeze_vision_encoder", False) and policy.actor.encoder.has_images:
                obs_feats = policy.actor.encoder.get_cached_image_features(observation)
            dist, _ = policy._actor_distribution(observation, obs_feats)
            action = dist.mode()
            q_values_current = policy.critic_ensemble(observation, action, obs_feats)
            q_min_current = q_values_current.min(dim=0)[0]
        
        valid_mask = torch.isfinite(q_min_current)
        if valid_mask.any():
            total_q_sum_current += q_min_current[valid_mask].sum().item()

        new_transition = step_lwlab_env_and_process_transition(
            env=online_env,
            transition=transition,
            action=action,
            env_processor=env_processor,
            action_processor=action_processor,
        )

        # Teleop action is the action that was executed in the environment
        # It is either the action from the teleop device or the action from the policy
        executed_action = new_transition[TransitionKey.ACTION]
        reward = new_transition[TransitionKey.REWARD]
        done = new_transition.get(TransitionKey.DONE, torch.tensor(False))
        truncated = new_transition.get(TransitionKey.TRUNCATED, torch.tensor(False))
        info = new_transition.get(TransitionKey.INFO, {})

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
            # make sure those will not be used!! (only create to use processer)
            del new_transition, new_transition_raw

            info.pop('final_obs') # remove final_obs from info to save space
            
        # assign obs to the next obs and continue the rollout
        if torch.any(done) or torch.any(truncated):
            transition = new_transition_with_reset
        else:
            transition = new_transition

    
    # Compute AM-Q score as mean of q_min values
    if total_samples > 0:
        score_prev = total_q_sum_prev / total_samples
        score_current = total_q_sum_current / total_samples
    else:
        score_prev = float("-inf")
        score_current = float("-inf")

    threshold = score_prev * (1 + adaptive_threshold_fraction)
    improvement = (score_current - score_prev) / score_prev
    if score_current > threshold:
        return score_current, total_samples, improvement, True
    else:
        return score_prev, total_samples, improvement, False