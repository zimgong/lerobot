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

from dataclasses import dataclass
from typing import Any, Sequence

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import einops

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.processor.pipeline import ProcessorStep, ProcessorStepRegistry, ObservationProcessorStep
from lerobot.processor.core import TransitionKey
from lerobot.utils.transition import Transition
from lerobot.processor.core import EnvTransition, EnvAction
from lerobot.processor.converters import to_tensor
from lerobot.utils.constants import OBS_ENV_STATE, OBS_IMAGE, OBS_IMAGES, OBS_STATE



@dataclass
@ProcessorStepRegistry.register(name="repack_lwlab_observation_processor")
class RepackLwlabObservationProcessorStep(ObservationProcessorStep):
    """
    Repacks LwLab environment observations to gym-style format for compatibility with VanillaObservationProcessorStep.
    
    This processor step converts the LwLab environment's observation format to a standard gym format that can be
    processed by the VanillaObservationProcessorStep. It handles:
    - Camera images: converts to "pixels" format
    - Environment state: converts to "environment_state" format  
    - Observation state: converts to "agent_pos" format
    
    This is equivalent to the original LwLabObservationProcessorWrapper but adapted for the new processor framework.
    """

    ENV_STATE_KEYS: list[str] | None = None
    OBS_STATE_KEYS: list[str] | None = None  
    CAMERA_KEYS: list[str] | None = None
    features: dict[str, Any] | None = None

    def __post_init__(self):
        """Initialize default values if not provided."""
        if self.ENV_STATE_KEYS is None:
            self.ENV_STATE_KEYS = []
        if self.OBS_STATE_KEYS is None:
            self.OBS_STATE_KEYS = ["joint_pos"]
        if self.CAMERA_KEYS is None:
            self.CAMERA_KEYS = None

    def _process_single_image(self, img: torch.Tensor, imgkey: str) -> torch.Tensor:
        """
        Processes a single image tensor, handling resizing if needed.
        
        Args:
            img: Input image tensor in channel-last format (B, H, W, C)
            imgkey: The key for this image (used for feature lookup)
            
        Returns:
            Processed image tensor in channel-first format (B, C, H, W)
        """
        # When preprocessing observations in a non-vectorized environment, we need to add a batch dimension.
        # This is the case for human-in-the-loop RL where there is only one environment.
        if img.ndim == 3:
            img = img.unsqueeze(0)
            
        # sanity check that images are channel last
        _, h, w, c = img.shape
        assert c < h and c < w, f"expect channel last images, but instead got {img.shape=}"

        # sanity check that images are uint8
        assert img.dtype == torch.uint8, f"expect torch.uint8, but instead {img.dtype=}"

        # convert to channel first of type float32 in range [0,1]
        img = einops.rearrange(img, "b h w c -> b c h w").contiguous()
        img = img.type(torch.float32)
        img /= 255

        if self.features and imgkey in self.features:
            # resize to match the target feature shape (width and height only)
            target_shape = self.features[imgkey].shape
            batch_size, c, h, w = img.shape
            
            # Extract target height and width (assuming shape is [C, H, W] or [H, W, C])
            if len(target_shape) >= 2:
                if len(target_shape) == 3:  # [C, H, W] format
                    target_h, target_w = target_shape[1], target_shape[2]
                else:  # [H, W] format
                    target_h, target_w = target_shape[0], target_shape[1]
                
                # Only resize if dimensions are different
                if h != target_h or w != target_w:
                    img = F.interpolate(
                        img, 
                        size=(target_h, target_w), 
                        mode='bilinear', 
                        align_corners=False
                    )
            else:
                # Fallback to reshape for non-image features
                img = img.reshape(batch_size, *target_shape)

        return img

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        """
        Convert LwLab environment observation to gym-style format.
        
        Args:
            observation: Dictionary of observation batches from a Gym vector environment.
            
        Returns:
            Dictionary of observation batches with keys renamed to gym format.
        """
        # Extract policy observations
        policy_obs = observation.get("policy", {})
        general_obs = observation.get("embodiment_general_obs", {})
        processed_obs = {}
        
        # Process camera images
        if self.CAMERA_KEYS is not None:
            # Use specified camera keys
            for key in self.CAMERA_KEYS:
                if key in policy_obs:
                    processed_obs[f"{OBS_IMAGES}.{key}"] = self._process_single_image(policy_obs[key], f"{OBS_IMAGES}.{key}")
        else:
            # Use all camera keys
            for key, img in policy_obs.items():
                if "camera" in key or "image" in key:
                    processed_obs[f"{OBS_IMAGES}.{key}"] = self._process_single_image(img, f"{OBS_IMAGES}.{key}")

        # Add environment state
        if self.ENV_STATE_KEYS:
            env_state_parts = []
            for key in self.ENV_STATE_KEYS:
                if key in policy_obs:
                    # fill nan with 0
                    policy_obs[key] = torch.where(torch.isnan(policy_obs[key]), torch.randn_like(policy_obs[key]), policy_obs[key])
                    env_state_parts.append(policy_obs[key])
                if key in general_obs:
                    general_obs[key] = torch.where(torch.isnan(general_obs[key]), torch.randn_like(general_obs[key]), general_obs[key])
                    env_state_parts.append(general_obs[key])
            if env_state_parts:
                env_state = torch.concat(env_state_parts, dim=1)
                processed_obs[OBS_ENV_STATE] = env_state

        # Add agent position (observation state)
        if self.OBS_STATE_KEYS:
            obs_state_parts = []
            for key in self.OBS_STATE_KEYS:
                if key in policy_obs:
                    # fill nan with 0
                    policy_obs[key] = torch.where(torch.isnan(policy_obs[key]), torch.randn_like(policy_obs[key]), policy_obs[key])
                    obs_state_parts.append(policy_obs[key])
                if key in general_obs:
                    general_obs[key] = torch.where(torch.isnan(general_obs[key]), torch.randn_like(general_obs[key]), general_obs[key])
                    obs_state_parts.append(general_obs[key])
            if obs_state_parts:
                obs_state = torch.concat(obs_state_parts, dim=1)
                processed_obs[OBS_STATE] = obs_state

        return processed_obs

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        Transforms feature keys from LwLab format to gym format.
        
        This method converts LwLab-specific feature keys to the standard gym format
        that can be processed by VanillaObservationProcessorStep.
        """
        # Build a new features mapping keyed by the same FeatureType buckets
        new_features: dict[PipelineFeatureType, dict[str, PolicyFeature]] = {ft: {} for ft in features.keys()}

        # Convert LwLab observation format to gym format
        for src_ft, bucket in features.items():
            for key, feat in bucket.items():
                # Handle camera images
                if "observation.images." in key:
                    # Convert to pixels format
                    camera_name = key.replace("observation.images.", "")
                    new_key = f"pixels.{camera_name}"
                    new_features[src_ft][new_key] = feat
                elif key == "observation.environment_state":
                    # Keep as is
                    new_key = "environment_state"
                    new_features[src_ft][new_key] = feat
                elif key == "observation.state":
                    # Convert to agent_pos
                    new_key = "agent_pos"
                    new_features[src_ft][new_key] = feat
                else:
                    # Keep other keys as is
                    new_features[src_ft][key] = feat

        return new_features


@dataclass
@ProcessorStepRegistry.register(name="lwlab_sparse_reward_processor")
class LwlabSparseRewardProcessorStep(ProcessorStep):
    """
    Processor step that implements sparse reward logic for LwLab environments.
    
    This processor step converts the reward to a sparse 0-1 reward based on the success status
    and handles the termination/truncation logic. This is equivalent to the original 
    LwlabSparseRewardWrapper but adapted for the new processor framework.
    """

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """
        Apply sparse reward logic to the transition.
        
        Args:
            transition: The input transition dictionary.
            
        Returns:
            The processed transition with sparse reward applied.
        """
        
        # Get the info dictionary
        info = transition.get(TransitionKey.INFO, {})
        
        # Apply sparse reward: 1.0 if success, 0.0 otherwise
        if "is_success" in info:
            is_success = info['is_success']
            if isinstance(is_success, torch.Tensor):
                reward = torch.where(is_success, 1.0, 0.0)
            else:
                reward = 1.0 if is_success else 0.0
            
            transition[TransitionKey.REWARD] = reward
        else:
            print("Warning: is_success is not in info")
        
        # Handle termination/truncation logic
        # Set truncated to True if either truncated or terminated is True
        assert TransitionKey.DONE in transition and TransitionKey.TRUNCATED in transition, "DONE and TRUNCATED must be in transition"
        terminated = transition[TransitionKey.DONE]
        truncated = transition[TransitionKey.TRUNCATED]

        if isinstance(terminated, torch.Tensor) and isinstance(truncated, torch.Tensor):
            new_done = torch.logical_or(truncated, terminated)
        else:
            new_done = truncated or terminated
            
        transition[TransitionKey.DONE] = new_done
        
        return transition

    def get_config(self) -> dict[str, Any]:
        """Returns the configuration of the step for serialization."""
        return {}
    
    def transform_features(self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Transforms the features of the step."""
        return features


# Action Processor Steps

@ProcessorStepRegistry.register("lwlab_numpy2torch_action_processor")
@dataclass
class LwlabNumpy2TorchActionProcessorStep(ProcessorStep):
    """Converts a NumPy array action to a PyTorch tensor when action is present."""

    device: str = "cuda:0"

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Converts numpy action to torch tensor if action exists, otherwise passes through."""
        from lerobot.processor.core import TransitionKey

        # if action is already a torch tensor, return the transition
        if isinstance(transition.get(TransitionKey.ACTION), torch.Tensor):
            return transition

        self._current_transition = transition.copy()
        new_transition = self._current_transition

        action = new_transition.get(TransitionKey.ACTION)
        if action is not None:
            if not isinstance(action, EnvAction):
                raise TypeError(
                    f"Expected np.ndarray or None, got {type(action).__name__}. "
                    "Use appropriate processor for non-tensor actions."
                )
            torch_action = to_tensor(action, dtype=None, device=self.device)  # Preserve original dtype
            new_transition[TransitionKey.ACTION] = torch_action

        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
