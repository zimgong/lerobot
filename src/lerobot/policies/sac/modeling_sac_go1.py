#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team.
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

import logging
import math
from collections.abc import Callable
from dataclasses import asdict
from typing import Literal

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor
from transformers import AutoTokenizer

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.sac.configuration_sac_go1 import SACGO1Config
from lerobot.policies.sac.modeling_sac import MLP, orthogonal_init, TanhMultivariateNormalDiag
from lerobot.policies.utils import get_device_from_parameters, get_dtype_from_parameters
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE

from go1.configs.go1_base_cfg import BaseSpaceArguments
from go1.internvl.model.go1 import GO1Model, GO1ModelConfig
from go1.internvl.train.constants import (
    BOX_END_TOKEN,
    BOX_START_TOKEN,
    IMG_CONTEXT_TOKEN,
    IMG_END_TOKEN,
    IMG_START_TOKEN,
    QUAD_END_TOKEN,
    QUAD_START_TOKEN,
    REF_END_TOKEN,
    REF_START_TOKEN,
)
from go1.internvl.train.dataset import build_transform
from go1.internvl.train.go1_train import build_ae_config, build_noise_scheduler_config
from go1.lerobot.dataset_lerobot import tensor_to_pil, WrappedLeRobotDataset
from go1.lerobot.dataset_transforms import make_conversation
from go1.tools.env_parse import get_bool_env

DISCRETE_DIMENSION_INDEX = -1  # Gripper is always the last dimension
logger = logging.getLogger(__name__)


class SACGO1Policy(
    PreTrainedPolicy,
):
    config_class = SACGO1Config
    name = "sac_go1"

    def __init__(
        self,
        config: SACGO1Config | None = None,
    ):
        super().__init__(config)
        config.validate_features()
        self.config = config

        # Determine action dimension and initialize all components
        space_dim = config.input_features[OBS_STATE].shape[0]
        continuous_action_dim = config.output_features[ACTION].shape[0]
        self._init_go1_model(space_dim, continuous_action_dim)
        self._init_encoders()
        self._init_critics(continuous_action_dim)
        self._init_actor(continuous_action_dim)
        self._init_temperature()

    def get_optim_params(self) -> dict:
        optim_params = {
            "actor": [
                p
                for n, p in self.actor.named_parameters()
                if not n.startswith("encoder") or not self.shared_encoder
            ],
            "critic": self.critic_ensemble.parameters(),
            "temperature": self.log_alpha,
        }
        if self.config.num_discrete_actions is not None:
            optim_params["discrete_critic"] = self.discrete_critic.parameters()
        return optim_params

    def reset(self):
        """Reset the policy"""
        pass

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        raise NotImplementedError("SACPolicy does not support action chunking. It returns single actions!")

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select action for inference/evaluation"""

        observations_features = None
        if self.shared_encoder:
            observations_features = self.actor.encoder.get_cached_image_features(batch)

        actions, _, _ = self.actor(batch, observations_features["vlm_outputs"])

        if self.config.num_discrete_actions is not None:
            discrete_action_value = self.discrete_critic(batch, observations_features["vlm_features"])
            discrete_action = torch.argmax(discrete_action_value, dim=-1, keepdim=True)
            actions = torch.cat([actions[:,0,:], discrete_action], dim=-1)

        return actions

    def critic_forward(
        self,
        observations: dict[str, Tensor],
        actions: Tensor,
        use_target: bool = False,
        observation_features: Tensor | None = None,
    ) -> Tensor:
        """Forward pass through a critic network ensemble

        Args:
            observations: Dictionary of observations
            actions: Action tensor
            use_target: If True, use target critics, otherwise use ensemble critics

        Returns:
            Tensor of Q-values from all critics
        """

        critics = self.critic_target if use_target else self.critic_ensemble
        q_values = critics(observations, actions, observation_features)
        return q_values

    def discrete_critic_forward(
        self, observations, use_target=False, observation_features=None
    ) -> torch.Tensor:
        """Forward pass through a discrete critic network

        Args:
            observations: Dictionary of observations
            use_target: If True, use target critics, otherwise use ensemble critics
            observation_features: Optional pre-computed observation features to avoid recomputing encoder output

        Returns:
            Tensor of Q-values from the discrete critic network
        """
        discrete_critic = self.discrete_critic_target if use_target else self.discrete_critic
        q_values = discrete_critic(observations, observation_features)
        return q_values

    def forward(
        self,
        batch: dict[str, Tensor | dict[str, Tensor]],
        model: Literal["actor", "critic", "temperature", "discrete_critic"] = "critic",
    ) -> dict[str, Tensor]:
        """Compute the loss for the given model

        Args:
            batch: Dictionary containing:
                - action: Action tensor
                - reward: Reward tensor
                - state: Observations tensor dict
                - next_state: Next observations tensor dict
                - done: Done mask tensor
                - observation_feature: Optional pre-computed observation features
                - next_observation_feature: Optional pre-computed next observation features
            model: Which model to compute the loss for ("actor", "critic", "discrete_critic", or "temperature")

        Returns:
            The computed loss tensor
        """
        # Extract common components from batch
        actions: Tensor = batch[ACTION]
        observations: dict[str, Tensor] = batch["state"]
        observation_features: Tensor = batch.get("observation_feature")

        if model == "critic":
            # Extract critic-specific components
            rewards: Tensor = batch["reward"]
            next_observations: dict[str, Tensor] = batch["next_state"]
            done: Tensor = batch["done"]
            next_observation_features: Tensor = batch.get("next_observation_feature")

            loss_critic = self.compute_loss_critic(
                observations=observations,
                actions=actions,
                rewards=rewards,
                next_observations=next_observations,
                done=done,
                observation_features=observation_features,
                next_observation_features=next_observation_features,
            )

            return {"loss_critic": loss_critic}

        if model == "discrete_critic" and self.config.num_discrete_actions is not None:
            # Extract critic-specific components
            rewards: Tensor = batch["reward"]
            next_observations: dict[str, Tensor] = batch["next_state"]
            done: Tensor = batch["done"]
            next_observation_features: Tensor = batch.get("next_observation_feature")
            complementary_info = batch.get("complementary_info")
            loss_discrete_critic = self.compute_loss_discrete_critic(
                observations=observations,
                actions=actions,
                rewards=rewards,
                next_observations=next_observations,
                done=done,
                observation_features=observation_features,
                next_observation_features=next_observation_features,
                complementary_info=complementary_info,
            )
            return {"loss_discrete_critic": loss_discrete_critic}
        if model == "actor":
            return {
                "loss_actor": self.compute_loss_actor(
                    observations=observations,
                    observation_features=observation_features,
                )
            }

        if model == "temperature":
            return {
                "loss_temperature": self.compute_loss_temperature(
                    observations=observations,
                    observation_features=observation_features,
                )
            }

        raise ValueError(f"Unknown model type: {model}")

    def update_target_networks(self):
        """Update target networks with exponential moving average"""
        for target_param, param in zip(
            self.critic_target.parameters(),
            self.critic_ensemble.parameters(),
            strict=True,
        ):
            target_param.data.copy_(
                param.data * self.config.critic_target_update_weight
                + target_param.data * (1.0 - self.config.critic_target_update_weight)
            )
        if self.config.num_discrete_actions is not None:
            for target_param, param in zip(
                self.discrete_critic_target.parameters(),
                self.discrete_critic.parameters(),
                strict=True,
            ):
                target_param.data.copy_(
                    param.data * self.config.critic_target_update_weight
                    + target_param.data * (1.0 - self.config.critic_target_update_weight)
                )

    def update_temperature(self):
        self.temperature = self.log_alpha.exp().item()

    def compute_loss_critic(
        self,
        observations,
        actions,
        rewards,
        next_observations,
        done,
        observation_features: Tensor | None = None,
        next_observation_features: Tensor | None = None,
    ) -> Tensor:
        with torch.no_grad():
            next_action_preds, next_log_probs, _ = self.actor(next_observations, next_observation_features)

            # 2- compute q targets
            q_targets = self.critic_forward(
                observations=next_observations,
                actions=next_action_preds,
                use_target=True,
                observation_features=next_observation_features,
            )

            # subsample critics to prevent overfitting if use high UTD (update to date)
            # TODO: Get indices before forward pass to avoid unnecessary computation
            if self.config.num_subsample_critics is not None:
                indices = torch.randperm(self.config.num_critics)
                indices = indices[: self.config.num_subsample_critics]
                q_targets = q_targets[indices]

            # critics subsample size
            min_q, _ = q_targets.min(dim=0)  # Get values from min operation
            if self.config.use_backup_entropy:
                min_q = min_q - (self.temperature * next_log_probs)

            td_target = rewards + (1 - done) * self.config.discount * min_q

        # 3- compute predicted qs
        if self.config.num_discrete_actions is not None:
            # NOTE: We only want to keep the continuous action part
            # In the buffer we have the full action space (continuous + discrete)
            # We need to split them before concatenating them in the critic forward
            actions: Tensor = actions[:, :DISCRETE_DIMENSION_INDEX]
        q_preds = self.critic_forward(
            observations=observations,
            actions=actions,
            use_target=False,
            observation_features=observation_features,
        )

        # 4- Calculate loss
        # Compute state-action value loss (TD loss) for all of the Q functions in the ensemble.
        td_target_duplicate = einops.repeat(td_target, "b -> e b", e=q_preds.shape[0])
        # You compute the mean loss of the batch for each critic and then to compute the final loss you sum them up
        critics_loss = (
            F.mse_loss(
                input=q_preds,
                target=td_target_duplicate,
                reduction="none",
            ).mean(dim=1)
        ).sum()
        return critics_loss

    def compute_loss_discrete_critic(
        self,
        observations,
        actions,
        rewards,
        next_observations,
        done,
        observation_features=None,
        next_observation_features=None,
        complementary_info=None,
    ):
        # NOTE: We only want to keep the discrete action part
        # In the buffer we have the full action space (continuous + discrete)
        # We need to split them before concatenating them in the critic forward
        actions_discrete: Tensor = actions[:, DISCRETE_DIMENSION_INDEX:].clone()
        actions_discrete = torch.round(actions_discrete)
        actions_discrete = actions_discrete.long()

        discrete_penalties: Tensor | None = None
        if complementary_info is not None:
            discrete_penalties: Tensor | None = complementary_info.get("discrete_penalty")

        with torch.no_grad():
            # For DQN, select actions using online network, evaluate with target network
            next_discrete_qs = self.discrete_critic_forward(
                next_observations, use_target=False, observation_features=next_observation_features
            )
            best_next_discrete_action = torch.argmax(next_discrete_qs, dim=-1, keepdim=True)

            # Get target Q-values from target network
            target_next_discrete_qs = self.discrete_critic_forward(
                observations=next_observations,
                use_target=True,
                observation_features=next_observation_features,
            )

            # Use gather to select Q-values for best actions
            target_next_discrete_q = torch.gather(
                target_next_discrete_qs, dim=1, index=best_next_discrete_action
            ).squeeze(-1)

            # Compute target Q-value with Bellman equation
            rewards_discrete = rewards
            if discrete_penalties is not None:
                rewards_discrete = rewards + discrete_penalties
            target_discrete_q = rewards_discrete + (1 - done) * self.config.discount * target_next_discrete_q

        # Get predicted Q-values for current observations
        predicted_discrete_qs = self.discrete_critic_forward(
            observations=observations, use_target=False, observation_features=observation_features
        )

        # Use gather to select Q-values for taken actions
        predicted_discrete_q = torch.gather(predicted_discrete_qs, dim=1, index=actions_discrete).squeeze(-1)

        # Compute MSE loss between predicted and target Q-values
        discrete_critic_loss = F.mse_loss(input=predicted_discrete_q, target=target_discrete_q)
        return discrete_critic_loss

    def compute_loss_temperature(self, observations, observation_features: Tensor | None = None) -> Tensor:
        """Compute the temperature loss"""
        # calculate temperature loss
        with torch.no_grad():
            _, log_probs, _ = self.actor(observations, observation_features)
        temperature_loss = (-self.log_alpha.exp() * (log_probs + self.target_entropy)).mean()
        return temperature_loss

    def compute_loss_actor(
        self,
        observations,
        observation_features: Tensor | None = None,
    ) -> Tensor:
        actions_pi, log_probs, _ = self.actor(observations, observation_features)

        q_preds = self.critic_forward(
            observations=observations,
            actions=actions_pi,
            use_target=False,
            observation_features=observation_features,
        )
        min_q_preds = q_preds.min(dim=0)[0]

        actor_loss = ((self.temperature * log_probs) - min_q_preds).mean()
        return actor_loss

    def _init_go1_model(self, space_dim, continuous_action_dim):
        """Initialize GO-1 VLM model and configure freezing."""
        model_args = self.config.go1_model_kwargs
        space_args = BaseSpaceArguments()
        space_args.state_dim = space_dim
        space_args.action_dim = continuous_action_dim
        space_args.space_repack = self.config.space_repack
        space_args.ctrl_freq = self.config.ctrl_freq

        # Load pretrained model, tokenizer, and image processor
        tokenizer_path = model_args.model_name_or_path or model_args.llm_path
        logger.info(f"Loading Tokenizer: {tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            add_eos_token=False,
            trust_remote_code=True,
            use_fast=model_args.use_fast_tokenizer,
        )
        tokenizer.tokenizer_path = tokenizer_path
        tokenizer.model_max_length = model_args.max_seq_length
        token_list = [
            IMG_START_TOKEN,
            IMG_END_TOKEN,
            IMG_CONTEXT_TOKEN,
            QUAD_START_TOKEN,
            QUAD_END_TOKEN,
            REF_START_TOKEN,
            REF_END_TOKEN,
            BOX_START_TOKEN,
            BOX_END_TOKEN,
        ]
        num_new_tokens = tokenizer.add_tokens(token_list, special_tokens=True)
        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

        # Determine training backend data dtype and model weights dtype
        torch_dtype = torch.bfloat16

        # Load model state dict from given model safetensor directory
        logger.info("Loading GO1Model...")
        # config = GO1ModelConfig.from_pretrained(model_args.model_name_or_path)
        config = GO1ModelConfig()
        config.llm_config.architectures = ["InternLM2ForCausalLMGO1"]
        config.vision_config.drop_path_rate = model_args.drop_path_rate
        config.llm_config.attn_implementation = "flash_attention_2"  # for InternLM
        config.pad_token_id = tokenizer.pad_token_id
        config.template = model_args.conv_style
        config.select_layer = model_args.vision_select_layer
        config.dynamic_image_size = model_args.dynamic_image_size
        config.use_thumbnail = model_args.use_thumbnail
        config.ps_version = model_args.ps_version
        config.min_dynamic_patch = model_args.min_dynamic_patch
        config.max_dynamic_patch = model_args.max_dynamic_patch
        config.img_context_token_id = img_context_token_id

        # rewrite the config for GO1
        ae_config = build_ae_config(model_args, config, space_args)
        config.action_config = ae_config
        config.action_chunk_size = ae_config.action_chunk_size
        noise_scheduler_config = build_noise_scheduler_config(model_args)
        config.noise_scheduler_config = noise_scheduler_config
        config.norm = True

        # Add latent planner related config
        if model_args.latent_planning:
            assert config.latent_planner_config.state_token_num == 0
            config.latent_planner_config.action_dim = 1  # codebook size is 32, and we need to do cross-entropy loss
            config.latent_planning = model_args.latent_planning

        model = GO1Model.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            torch_dtype=torch_dtype,
            _fast_init=get_bool_env(name="DEBUG_MODE"),
            ignore_mismatched_sizes=True,
        )

        assert model.config.downsample_ratio == model_args.down_sample_ratio
        patch_size = model.config.vision_config.patch_size
        logger.info(f"model.config.force_image_size: {model.config.force_image_size}")
        logger.info(f"model_args.force_image_size: {model_args.force_image_size}")
        logger.info(f"model.config.vision_config.image_size: {model.config.vision_config.image_size}")
        if model.config.vision_config.image_size != model_args.force_image_size:
            logger.info(
                f"Resizing position embedding from "
                f"{model.config.vision_config.image_size} "
                f"to {model_args.force_image_size}..."
            )
            model.vision_model.resize_pos_embeddings(
                old_size=model.config.vision_config.image_size,
                new_size=model_args.force_image_size,
                patch_size=patch_size,
            )
            model.config.vision_config.image_size = model_args.force_image_size
        model.config.force_image_size = model_args.force_image_size
        model.num_image_token = int((model_args.force_image_size // patch_size) ** 2 * (model_args.down_sample_ratio**2))

        if num_new_tokens > 0:
            model.language_model.resize_token_embeddings(len(tokenizer))
            output_embeddings = model.language_model.get_output_embeddings().weight.data
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
            output_embeddings[-num_new_tokens:] = output_embeddings_avg

            model.config.llm_config.vocab_size = len(tokenizer)
            model.language_model.config.vocab_size = len(tokenizer)

        model.language_model.config.use_cache = True
        model.vision_model.gradient_checkpointing = True
        model.vision_model.encoder.gradient_checkpointing = True
        model.language_model._set_output_logits(model_args.output_logits)
        if model_args.grad_checkpoint:
            model.language_model._set_gradient_checkpointing()
            model.latent_planner._set_gradient_checkpointing()
            model.action_model._set_gradient_checkpointing()

        # Model freeze params operation
        def _freeze_params(module):
            for param in module.parameters():
                param.requires_grad = False

        if model_args.freeze_backbone:
            _freeze_params(model.vision_model)

        if model_args.freeze_llm:
            model.language_model = model.language_model.eval()
            _freeze_params(model.language_model)

        if model_args.freeze_mlp:
            _freeze_params(model.mlp1)

        if model_args.freeze_latent_planner:
            _freeze_params(model.latent_planner)

        self.tokenizer = tokenizer
        self.go1_model = model

    def _init_encoders(self):
        """Initialize shared or separate encoders for actor and critic."""
        self.shared_encoder = self.config.shared_encoder
        self.encoder_critic = SACObservationEncoder(self.config, self.tokenizer, self.go1_model)
        self.encoder_actor = (
            self.encoder_critic if self.shared_encoder else SACObservationEncoder(self.config, self.tokenizer, self.go1_model)
        )

    def _init_critics(self, continuous_action_dim):
        """Build critic ensemble, targets, and optional discrete critic."""
        heads = [
            CriticHead(
                input_dim=self.encoder_critic.output_dim + continuous_action_dim,
                **asdict(self.config.critic_network_kwargs),
            )
            for _ in range(self.config.num_critics)
        ]
        self.critic_ensemble = CriticEnsemble(encoder=self.encoder_critic, ensemble=heads)
        target_heads = [
            CriticHead(
                input_dim=self.encoder_critic.output_dim + continuous_action_dim,
                **asdict(self.config.critic_network_kwargs),
            )
            for _ in range(self.config.num_critics)
        ]
        self.critic_target = CriticEnsemble(encoder=self.encoder_critic, ensemble=target_heads)
        self.critic_target.load_state_dict(self.critic_ensemble.state_dict())

        if self.config.use_torch_compile:
            self.critic_ensemble = torch.compile(self.critic_ensemble)
            self.critic_target = torch.compile(self.critic_target)

        if self.config.num_discrete_actions is not None:
            self._init_discrete_critics()

    def _init_discrete_critics(self):
        """Build discrete discrete critic ensemble and target networks."""
        self.discrete_critic = DiscreteCritic(
            encoder=self.encoder_critic,
            input_dim=self.encoder_critic.output_dim,
            output_dim=self.config.num_discrete_actions,
            **asdict(self.config.discrete_critic_network_kwargs),
        )
        self.discrete_critic_target = DiscreteCritic(
            encoder=self.encoder_critic,
            input_dim=self.encoder_critic.output_dim,
            output_dim=self.config.num_discrete_actions,
            **asdict(self.config.discrete_critic_network_kwargs),
        )

        # TODO: (maractingi, azouitine) Compile the discrete critic
        self.discrete_critic_target.load_state_dict(self.discrete_critic.state_dict())

    def _init_actor(self, continuous_action_dim):
        """Initialize policy actor network and default target entropy."""
        # NOTE: The actor select only the continuous action part
        self.actor = Policy(
            encoder=self.encoder_actor,
            network=self.go1_model,
            action_dim=continuous_action_dim,
            encoder_is_shared=self.shared_encoder,
            ctrl_freq=self.config.ctrl_freq,
            **asdict(self.config.policy_kwargs),
        )

        self.target_entropy = self.config.target_entropy
        if self.target_entropy is None:
            dim = continuous_action_dim + (1 if self.config.num_discrete_actions is not None else 0)
            self.target_entropy = -np.prod(dim) / 2

    def _init_temperature(self):
        """Set up temperature parameter and initial log_alpha."""
        temp_init = self.config.temperature_init
        self.log_alpha = nn.Parameter(torch.tensor([math.log(temp_init)]))
        self.temperature = self.log_alpha.exp().item()


class SACObservationEncoder(nn.Module):
    """Encode image and/or state vector observations.Encode VLM observations using last layer key-value pairs.
    
    This encoder extracts key-value pairs from the last transformer layer of the VLM
    and pools them to create a compact representation for the critic head. The design
    reduces the high-dimensional key-value cache to a manageable size while preserving
    the most relevant information from the VLM's final layer.
    
    The output dimension is 2 * head_dim, where head_dim is the dimension per attention
    head. This comes from concatenating the pooled key and value vectors.
    """

    def __init__(self, config: SACGO1Config, tokenizer, vla_model) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.vla_model = vla_model
        self.head_dim = self.vla_model.language_model.config.hidden_size // self.vla_model.language_model.config.num_attention_heads
        self.mlp_head = MLP(input_dim=2 * self.head_dim, hidden_dims=[4 * self.head_dim, 2 * self.head_dim], activate_final=True, final_activation=None)
        self._compute_output_dim()

    def _compute_output_dim(self) -> None:
        # Extract last layer key-value pairs for critic input
        # Each key/value has shape: (batch_size, num_heads, seq_len, head_dim)
        # We'll pool across sequence length and heads to get: (batch_size, head_dim)
        # Use both key and value from last layer, so 2 * head_dim
        # Then pass through MLP head which outputs 256 dimensions
        self._out_dim = 2 * self.head_dim  # Output dimension of the MLP head

    def forward(
        self, obs: dict[str, Tensor], cache: dict[str, Tensor] | None = None, detach: bool = False
    ):
        if cache is None:
            cache = self.get_cached_image_features(obs, detach)
        return cache

    def get_cached_image_features(self, obs: dict[str, Tensor], detach: bool = False) -> dict[str, Tensor]:
        """Extract and optionally cache image features from observations.

        This function processes image observations through the vision encoder once and returns
        the resulting features.
        When the image encoder is shared between actor and critics AND frozen, these features can be safely cached and
        reused across policy components (actor, critic, discrete_critic), avoiding redundant forward passes.

        Performance impact:
        - The vision encoder forward pass is typically the main computational bottleneck during training and inference
        - Caching these features can provide 2-4x speedup in training and inference

        Usage patterns:
        - Called in select_action()
        - Called in learner.py's get_observation_features() to pre-compute features for all policy components
        - Called internally by forward()

        Args:
            obs: Dictionary of observation tensors containing image keys

        Returns:
            Dictionary mapping image keys to their corresponding encoded features
        """
        if "cam_head_color" in self.config.space_repack:
            obs["cam_head_color"] = tensor_to_pil(
                obs[self.config.space_repack["cam_head_color"]].squeeze(0).permute(1, 2, 0)
            )
        if "cam_hand_right_color" in self.config.space_repack:
            obs["cam_hand_right_color"] = tensor_to_pil(
                obs[self.config.space_repack["cam_hand_right_color"]].squeeze(0).permute(1, 2, 0)
            )
        if "cam_hand_left_color" in self.config.space_repack:
            obs["cam_hand_left_color"] = tensor_to_pil(
                obs[self.config.space_repack["cam_hand_left_color"]].squeeze(0).permute(1, 2, 0)
            )
        if "final_prompt" in self.config.space_repack:
            obs["final_prompt"] = tensor_to_pil(
                obs[self.config.space_repack["final_prompt"]].squeeze(0).permute(1, 2, 0)
            )
        else:
            obs["final_prompt"] = self.config.default_prompt
        obs["final_prompt"] = make_conversation(prompt=obs["final_prompt"])

        transform = build_transform(
            is_train=True,
            input_size=self.vla_model.config.force_image_size,
            pad2square=self.vla_model.config.pad2square,
            normalize_type="imagenet"
        )
        observation_features = WrappedLeRobotDataset.multi_image_get_item(
            raw_target=obs,
            img_transform=transform,
            text_tokenizer=self.tokenizer,
            num_image_token=self.vla_model.num_image_token,
            use_thumbnail=self.vla_model.config.use_thumbnail,
            min_dynamic_patch=self.vla_model.config.min_dynamic_patch,
            max_dynamic_patch=self.vla_model.config.max_dynamic_patch,
            image_size=self.vla_model.config.force_image_size,
        )

        if "cam_head_color" in obs:
            obs.pop("cam_head_color")
        if "cam_hand_right_color" in obs:
            obs.pop("cam_hand_right_color")
        if "cam_hand_left_color" in obs:
            obs.pop("cam_hand_left_color")
        if "final_prompt" in obs:
            obs.pop("final_prompt")

        pixel_values = observation_features["pixel_values"].to(dtype=self.vla_model.dtype, device=self.vla_model.device)
        input_ids = observation_features["input_ids"].unsqueeze(0).to(self.vla_model.device)
        attention_mask = observation_features["attention_mask"].unsqueeze(0).to(self.vla_model.device)
        position_ids = observation_features["position_ids"].unsqueeze(0).to(self.vla_model.device)
        image_flags = observation_features["image_flags"].to(self.vla_model.device)
        labels = observation_features["labels"].unsqueeze(0).to(self.vla_model.device)

        vlm_outputs = self.vla_model.common_process(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            image_flags=image_flags,
            return_dict=None,
            labels=labels,
        )
        vlm_outputs.attention_mask = attention_mask
        if detach:
            vlm_outputs = vlm_outputs.detach()
        
        # Extract last layer key-value pairs
        # past_key_values is a tuple of (key, value) pairs for each layer
        last_layer_kv = vlm_outputs.past_key_values[-1]  # Get last layer
        last_key, last_value = last_layer_kv  # Unpack key and value
        
        # Apply attention mask to avoid pooling over padding tokens
        # attention_mask shape: (batch_size, seq_len)
        if attention_mask is not None:
            # Expand attention mask to match key/value dimensions
            # Shape: (batch_size, 1, seq_len, 1) for broadcasting
            mask_expanded = attention_mask.unsqueeze(1).unsqueeze(-1)
            
            # Apply mask and compute masked mean
            masked_key = last_key * mask_expanded
            masked_value = last_value * mask_expanded
            
            # Sum over sequence length and heads, then divide by number of valid tokens
            valid_tokens = mask_expanded.sum(dim=(1, 2), keepdim=True)  # (batch_size, 1, 1, 1)
            pooled_key = masked_key.sum(dim=(1, 2)) / (valid_tokens.squeeze(-1) + 1e-8)
            pooled_value = masked_value.sum(dim=(1, 2)) / (valid_tokens.squeeze(-1) + 1e-8)
        else:
            # Fallback to simple mean pooling if no attention mask
            pooled_key = last_key.mean(dim=(1, 2))  # Average over heads and sequence length
            pooled_value = last_value.mean(dim=(1, 2))  # Average over heads and sequence length
        
        # Concatenate key and value features
        # Shape: (batch_size, 2 * head_dim)
        kv_features = torch.cat([pooled_key, pooled_value], dim=-1).squeeze(1)
        
        # Pass through MLP head to get final features
        # Shape: (batch_size, 256)
        vlm_features = self.mlp_head(kv_features)

        return {"vlm_outputs": vlm_outputs, "vlm_features": vlm_features}

    def _encode_images(self, cache: dict[str, Tensor], detach: bool) -> Tensor:
        """Encode image features from cached observations.

        This function takes pre-encoded image features from the cache and applies spatial embeddings and post-encoders.
        It also supports detaching the encoded features if specified.

        Args:
            cache (dict[str, Tensor]): The cached image features.
            detach (bool): Usually when the encoder is shared between actor and critics,
            we want to detach the encoded features on the policy side to avoid backprop through the encoder.
            More detail here `https://cdn.aaai.org/ojs/17276/17276-13-20770-1-2-20210518.pdf`

        Returns:
            Tensor: The encoded image features.
        """
        pass

    @property
    def output_dim(self) -> int:
        return self._out_dim


class CriticHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        activations: Callable[[torch.Tensor], torch.Tensor] | str = nn.SiLU(),
        activate_final: bool = False,
        dropout_rate: float | None = None,
        init_final: float | None = None,
        final_activation: Callable[[torch.Tensor], torch.Tensor] | str | None = None,
    ):
        super().__init__()
        self.net = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            activations=activations,
            activate_final=activate_final,
            dropout_rate=dropout_rate,
            final_activation=final_activation,
        )
        self.output_layer = nn.Linear(in_features=hidden_dims[-1], out_features=1)
        if init_final is not None:
            nn.init.uniform_(self.output_layer.weight, -init_final, init_final)
            nn.init.uniform_(self.output_layer.bias, -init_final, init_final)
        else:
            orthogonal_init()(self.output_layer.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_layer(self.net(x))


class CriticEnsemble(nn.Module):
    """
    CriticEnsemble wraps multiple CriticHead modules into an ensemble.

    Args:
        encoder (SACObservationEncoder): encoder for observations.
        ensemble (List[CriticHead]): list of critic heads.
        init_final (float | None): optional initializer scale for final layers.

    Forward returns a tensor of shape (num_critics, batch_size) containing Q-values.
    """

    def __init__(
        self,
        encoder: SACObservationEncoder,
        ensemble: list[CriticHead],
        init_final: float | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.init_final = init_final
        self.critics = nn.ModuleList(ensemble)

    def forward(
        self,
        observations: dict[str, torch.Tensor],
        actions: torch.Tensor,
        observation_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)
        # Move each tensor in observations to device and dtype
        observations = {k: v.to(device=device, dtype=dtype) for k, v in observations.items()}
        actions = actions.to(device=device, dtype=dtype)

        obs_enc = self.encoder(observations, cache=observation_features)

        inputs = torch.cat([obs_enc, actions], dim=-1)

        # Loop through critics and collect outputs
        q_values = []
        for critic in self.critics:
            q_values.append(critic(inputs))

        # Stack outputs to match expected shape [num_critics, batch_size]
        q_values = torch.stack([q.squeeze(-1) for q in q_values], dim=0)
        return q_values


class DiscreteCritic(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int = 3,
        activations: Callable[[torch.Tensor], torch.Tensor] | str = nn.SiLU(),
        activate_final: bool = False,
        dropout_rate: float | None = None,
        init_final: float | None = None,
        final_activation: Callable[[torch.Tensor], torch.Tensor] | str | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.output_dim = output_dim

        self.net = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            activations=activations,
            activate_final=activate_final,
            dropout_rate=dropout_rate,
            final_activation=final_activation,
        )

        self.output_layer = nn.Linear(in_features=hidden_dims[-1], out_features=self.output_dim)
        if init_final is not None:
            nn.init.uniform_(self.output_layer.weight, -init_final, init_final)
            nn.init.uniform_(self.output_layer.bias, -init_final, init_final)
        else:
            orthogonal_init()(self.output_layer.weight)

    def forward(
        self, observations: torch.Tensor, observation_features: torch.Tensor | None = None
    ) -> torch.Tensor:
        obs_enc = self.encoder(observations, cache=observation_features)
        return self.output_layer(self.net(obs_enc))


class Policy(nn.Module):
    def __init__(
        self,
        encoder: SACObservationEncoder,
        network: nn.Module,
        action_dim: int,
        std_min: float = -5,
        std_max: float = 2,
        fixed_std: torch.Tensor | None = None,
        init_final: float | None = None,
        use_tanh_squash: bool = False,
        encoder_is_shared: bool = False,
        ctrl_freq = None,
    ):
        super().__init__()
        self.encoder: SACObservationEncoder = encoder
        self.network = network
        self.action_dim = action_dim
        self.std_min = std_min
        self.std_max = std_max
        self.fixed_std = fixed_std
        self.use_tanh_squash = use_tanh_squash
        self.encoder_is_shared = encoder_is_shared
        self.ctrl_freq = ctrl_freq
        # Find the last Linear layer's output dimension
        out_features = self.network.config.action_config.hidden_size
        # Mean layer
        self.mean_layer = nn.Linear(out_features, action_dim)
        if init_final is not None:
            nn.init.uniform_(self.mean_layer.weight, -init_final, init_final)
            nn.init.uniform_(self.mean_layer.bias, -init_final, init_final)
        else:
            orthogonal_init()(self.mean_layer.weight)

        # Standard deviation layer or parameter
        if fixed_std is None:
            self.std_layer = nn.Linear(out_features, action_dim)
            if init_final is not None:
                nn.init.uniform_(self.std_layer.weight, -init_final, init_final)
                nn.init.uniform_(self.std_layer.bias, -init_final, init_final)
            else:
                orthogonal_init()(self.std_layer.weight)

    def forward(
        self,
        observations: torch.Tensor,
        observation_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # We detach the encoder if it is shared to avoid backprop through it
        # This is important to avoid the encoder to be updated through the policy
        attention_mask = observation_features.attention_mask
        obs_enc = self.encoder(observations, cache=observation_features, detach=self.encoder_is_shared)

        # Get network outputs
        state = observations[OBS_STATE].to(dtype=self.network.dtype).unsqueeze(1)
        B = state.shape[0]
        ctrl_freqs = torch.full((B, 1), self.ctrl_freq, dtype=self.network.dtype, device=state.device)

        # Project vlm kv_cache head dim into action expert head dim
        vlm_key_values = obs_enc.past_key_values  # kv_cache(multi_head) of each decoder layer

        vlm_key_values_downsample = []
        for vlm_key_value, k_proj, v_proj in zip(vlm_key_values, self.network.k_proj_layers, self.network.v_proj_layers):
            vlm_key_values_downsample.append((k_proj(vlm_key_value[0]), v_proj(vlm_key_value[1])))

        if self.network.enable_lam:
            (
                latent_vlm_key_values_downsample,
                outputs_latent,
            ) = self.network.latent_planner(
                vlm_key_values_downsample,
                attention_mask,
            )

        # Sample noise that we'll add to the actions
        dummy_action = torch.zeros(B, self.network.config.action_chunk_size, self.network.action_dim, dtype=self.network.dtype, device=state.device)
        noise = torch.randn(dummy_action.shape, dtype=self.network.dtype, device=state.device)
        timesteps = torch.randint(0, self.network.num_train_timesteps, (B, 1), device=state.device).long()
        timestep_tokens = self.network.time_embedder(timesteps)
        freq_tokens = self.network.freq_embedder(ctrl_freqs)
        noisy_action = self.network.noise_scheduler.add_noise(dummy_action, noise, timesteps)

        state_trajs = self.network.state_adaptor(state)
        action_trajs = self.network.action_adaptor(noisy_action)
        state_action_trajs = torch.cat([state_trajs, action_trajs], dim=1)

        state_action_trajs_w_tfps = torch.cat(
            [timestep_tokens, freq_tokens, state_action_trajs], dim=1
        )

        # Action expert as diffusion head
        if self.network.enable_lam:
            vlm_key_values_downsample = latent_vlm_key_values_downsample
            attention_mask = torch.cat(
                (
                    attention_mask,
                    torch.ones(
                        B, self.network.latent_planner.latent_token_nums, dtype=torch.bool, device=attention_mask.device
                    ),
                ),
                dim=1,
            )

        outputs = self.network.action_model(state_action_trajs_w_tfps, attention_mask, vlm_key_values_downsample)
        state_action_output_tokens = outputs[0].to(dtype=self.mean_layer.weight.dtype)
        action_output_tokens = state_action_output_tokens[:, -self.network.action_chunk_size :, ...]
        means = self.mean_layer(action_output_tokens)

        # Compute standard deviations
        if self.fixed_std is None:
            log_std = self.std_layer(action_output_tokens)
            std = torch.exp(log_std)  # Match JAX "exp"
            std = torch.clamp(std, self.std_min, self.std_max)  # Match JAX default clip
        else:
            std = self.fixed_std.expand_as(means)

        # Build transformed distribution
        dist = TanhMultivariateNormalDiag(loc=means, scale_diag=std)

        # Sample actions (reparameterized)
        actions = dist.rsample()

        # Compute log_probs
        log_probs = dist.log_prob(actions)

        return actions, log_probs, means

    def get_features(self, observations: torch.Tensor) -> torch.Tensor:
        """Get encoded features from observations"""
        return NotImplementedError("Not implemented")
