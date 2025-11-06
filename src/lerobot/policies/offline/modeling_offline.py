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

from __future__ import annotations

from dataclasses import asdict
from typing import Literal

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from lerobot.policies.offline.configuration_offline import OfflineIQLConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.sac.modeling_sac import (
    CriticEnsemble,
    CriticHead,
    DiscreteCritic,
    DISCRETE_DIMENSION_INDEX,
    MLP,
    Policy,
    SACObservationEncoder,
    TanhMultivariateNormalDiag,
)
from lerobot.utils.constants import ACTION


class OfflineIQLPolicy(PreTrainedPolicy):
    """Implicit Q-Learning (IQL) policy with Advantage Weighted Regression (AWR) actor."""

    config_class = OfflineIQLConfig
    name = "offline"

    def __init__(self, config: OfflineIQLConfig):
        super().__init__(config)
        config.validate_features()
        self.config = config

        continuous_action_dim = config.output_features[ACTION].shape[0]

        self._init_encoders()
        self._init_value_head()
        self._init_critics(continuous_action_dim)
        self._init_actor(continuous_action_dim)
        if config.num_discrete_actions is not None:
            self._init_discrete_critics()

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def get_optim_params(self) -> dict:
        """Collect parameter groups for the optimizers."""
        actor_params = [
            p
            for name, p in self.actor.named_parameters()
            if not self.shared_encoder or not name.startswith("encoder")
        ]
        optim_params = {
            "actor": actor_params,
            "critic": self.critic_ensemble.parameters(),
            "value": self.value_head.parameters(),
        }
        if self.config.num_discrete_actions is not None:
            optim_params["discrete_critic"] = self.discrete_critic.parameters()
        return optim_params

    def reset(self):
        """No stateful caches to clear for this policy."""
        pass

    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        raise NotImplementedError("OfflineIQLPolicy does not support action chunking.")

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        observation_features = None
        if self.shared_encoder and self.actor.encoder.has_images:
            observation_features = self.actor.encoder.get_cached_image_features(batch)
        dist, _ = self._actor_distribution(batch, observation_features)
        actions = dist.mode()
        
        # Handle discrete actions if configured
        if self.config.num_discrete_actions is not None:
            discrete_action_value = self.discrete_critic(batch, observation_features)
            discrete_action = torch.argmax(discrete_action_value, dim=-1, keepdim=True)
            actions = torch.cat([actions, discrete_action], dim=-1)
        
        return actions

    def forward(
        self,
        batch: dict[str, Tensor | dict[str, Tensor]],
        model: Literal["critic", "value", "actor", "actor_bc", "discrete_critic"] = "critic",
    ) -> dict[str, Tensor]:
        actions: Tensor = batch[ACTION]
        observations: dict[str, Tensor] = batch["state"]
        observation_features: Tensor | None = batch.get("observation_feature")

        if model == "critic":
            rewards: Tensor = batch["reward"]
            next_observations: dict[str, Tensor] = batch["next_state"]
            next_observation_features: Tensor | None = batch.get("next_observation_feature")
            done: Tensor = batch["done"]

            critic_loss, q_info = self.compute_loss_critic(
                observations=observations,
                actions=actions,
                rewards=rewards,
                next_observations=next_observations,
                done=done,
                observation_features=observation_features,
                next_observation_features=next_observation_features,
                return_q_info=True,
            )
            return {"loss_critic": critic_loss, "q_info": q_info}

        if model == "value":
            return {
                "loss_value": self.compute_loss_value(
                    observations=observations,
                    actions=actions,
                    observation_features=observation_features,
                )
            }

        if model == "actor":
            actor_loss, actor_info = self.compute_loss_actor(
                observations=observations,
                actions=actions,
                observation_features=observation_features,
            )
            return {"loss_actor": actor_loss, "actor_info": actor_info}

        if model == "actor_bc":
            bc_loss, bc_info = self.compute_loss_actor_bc(
                observations=observations,
                actions=actions,
            )
            return {"loss_actor_bc": bc_loss, "actor_bc_info": bc_info}

        if model == "discrete_critic" and self.config.num_discrete_actions is not None:
            rewards: Tensor = batch["reward"]
            next_observations: dict[str, Tensor] = batch["next_state"]
            next_observation_features: Tensor | None = batch.get("next_observation_feature")
            done: Tensor = batch["done"]
            complementary_info = batch.get("complementary_info")
            loss_discrete_critic, q_info = self.compute_loss_discrete_critic(
                observations=observations,
                actions=actions,
                rewards=rewards,
                next_observations=next_observations,
                done=done,
                observation_features=observation_features,
                next_observation_features=next_observation_features,
                complementary_info=complementary_info,
                return_q_info=True,
            )
            return {"loss_discrete_critic": loss_discrete_critic, "q_info": q_info}

        raise ValueError(f"Unknown model type: {model}")

    # --------------------------------------------------------------------- #
    # Initialization helpers
    # --------------------------------------------------------------------- #
    def _init_encoders(self) -> None:
        self.shared_encoder = self.config.shared_encoder
        self.encoder_critic = SACObservationEncoder(self.config)
        self.encoder_actor = self.encoder_critic if self.shared_encoder else SACObservationEncoder(self.config)

    def _init_value_head(self) -> None:
        critic_kwargs = asdict(self.config.critic_network_kwargs)
        self.value_head = CriticHead(
            input_dim=self.encoder_critic.output_dim,
            **critic_kwargs,
        )
        self.value_target_head = CriticHead(
            input_dim=self.encoder_critic.output_dim,
            **critic_kwargs,
        )
        self.value_target_head.load_state_dict(self.value_head.state_dict())

    def _init_critics(self, continuous_action_dim: int) -> None:
        critic_kwargs = asdict(self.config.critic_network_kwargs)
        heads = [
            CriticHead(
                input_dim=self.encoder_critic.output_dim + continuous_action_dim,
                **critic_kwargs,
            )
            for _ in range(self.config.num_critics)
        ]
        self.critic_ensemble = CriticEnsemble(encoder=self.encoder_critic, ensemble=heads)

        target_heads = [
            CriticHead(
                input_dim=self.encoder_critic.output_dim + continuous_action_dim,
                **critic_kwargs,
            )
            for _ in range(self.config.num_critics)
        ]
        self.critic_target = CriticEnsemble(encoder=self.encoder_critic, ensemble=target_heads)
        self.critic_target.load_state_dict(self.critic_ensemble.state_dict())

    def _init_actor(self, continuous_action_dim: int) -> None:
        actor_kwargs = asdict(self.config.actor_network_kwargs)
        policy_kwargs = asdict(self.config.policy_kwargs)
        self.actor = Policy(
            encoder=self.encoder_actor,
            network=MLP(
                input_dim=self.encoder_actor.output_dim,
                **actor_kwargs,
            ),
            action_dim=continuous_action_dim,
            encoder_is_shared=self.shared_encoder,
            **policy_kwargs,
        )

    def _init_discrete_critics(self) -> None:
        """Build discrete critic and target networks."""
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
        self.discrete_critic_target.load_state_dict(self.discrete_critic.state_dict())

    # --------------------------------------------------------------------- #
    # Critic / Value losses
    # --------------------------------------------------------------------- #
    def critic_forward(
        self,
        observations: dict[str, Tensor],
        actions: Tensor,
        use_target: bool = False,
        observation_features: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        critics = self.critic_target if use_target else self.critic_ensemble
        q_values = critics(observations, actions, observation_features)
        q_mean = q_values.mean(dim=0)
        q_std = q_values.std(dim=0)
        return q_values, q_mean, q_std

    def discrete_critic_forward(
        self,
        observations: dict[str, Tensor],
        use_target: bool = False,
        observation_features: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Forward pass through a discrete critic network.

        Args:
            observations: Dictionary of observations
            use_target: If True, use target discrete critic, otherwise use online discrete critic
            observation_features: Optional pre-computed observation features

        Returns:
            Tuple of (q_values, q_mean, q_std) where:
            - q_values: Tensor of Q-values from the discrete critic network [batch_size, num_discrete_actions]
            - q_mean: Mean Q-value across actions
            - q_std: Standard deviation of Q-values across actions
        """
        discrete_critic = self.discrete_critic_target if use_target else self.discrete_critic
        q_values = discrete_critic(observations, observation_features)
        q_mean = q_values.mean(dim=-1)
        q_std = q_values.std(dim=-1)
        return q_values, q_mean, q_std

    def value_forward(
        self,
        observations: dict[str, Tensor],
        observation_features: Tensor | None = None,
        use_target: bool = False,
        detach_encoder: bool = True,
    ) -> Tensor:
        head = self.value_target_head if use_target else self.value_head
        features = self.encoder_critic(
            observations,
            cache=observation_features,
            detach=detach_encoder,
        )
        values = head(features).squeeze(-1)
        return values

    def compute_loss_value(
        self,
        observations: dict[str, Tensor],
        actions: Tensor,
        observation_features: Tensor | None = None,
    ) -> Tensor:
        # NOTE: For discrete actions, we only use the continuous action part
        if self.config.num_discrete_actions is not None:
            actions: Tensor = actions[:, :DISCRETE_DIMENSION_INDEX]
        
        with torch.no_grad():
            target_q = self.critic_target(observations, actions, observation_features)
            target_q = target_q.min(dim=0)[0]
        values = self.value_forward(
            observations=observations,
            observation_features=observation_features,
            use_target=False,
            detach_encoder=True,
        )
        delta = target_q - values
        tau = self.config.expectile_tau
        weight = torch.where(delta >= 0, tau, 1 - tau)
        return (weight * delta.square()).mean()

    def compute_loss_critic(
        self,
        observations: dict[str, Tensor],
        actions: Tensor,
        rewards: Tensor,
        next_observations: dict[str, Tensor],
        done: Tensor,
        observation_features: Tensor | None = None,
        next_observation_features: Tensor | None = None,
        return_q_info: bool = False,
    ) -> tuple[Tensor, dict] | Tensor:
        # NOTE: For discrete actions, we only use the continuous action part
        # In the buffer we have the full action space (continuous + discrete)
        # We need to split them before concatenating them in the critic forward
        if self.config.num_discrete_actions is not None:
            actions: Tensor = actions[:, :DISCRETE_DIMENSION_INDEX]
        
        with torch.no_grad():
            target_values = self.value_forward(
                observations=next_observations,
                observation_features=next_observation_features,
                use_target=True,
                detach_encoder=True,
            )
            target = rewards + self.config.discount * (1.0 - done.float()) * target_values
        q_values = self.critic_ensemble(observations, actions, observation_features)
        target_expanded = target.unsqueeze(0)
        loss = F.mse_loss(q_values, target_expanded)
        if not return_q_info:
            return loss
        q_info = {
            "q_mean": q_values.mean().item(),
            "q_std": q_values.std().item(),
            "target_mean": target.mean().item(),
        }
        return loss, q_info

    # --------------------------------------------------------------------- #
    # Actor loss
    # --------------------------------------------------------------------- #
    def compute_loss_actor(
        self,
        observations: dict[str, Tensor],
        actions: Tensor,
        observation_features: Tensor | None = None,
    ) -> tuple[Tensor, dict]:
        # NOTE: For discrete actions, we only use the continuous action part
        if self.config.num_discrete_actions is not None:
            actions: Tensor = actions[:, :DISCRETE_DIMENSION_INDEX]
        
        dist, _ = self._actor_distribution(observations, observation_features)
        log_prob = dist.log_prob(actions)

        with torch.no_grad():
            q_values = self.critic_ensemble(observations, actions, observation_features)
            q_min = q_values.min(dim=0)[0]
            v_values = self.value_forward(
                observations=observations,
                observation_features=observation_features,
                use_target=False,
                detach_encoder=True,
            )
            advantage = q_min - v_values
            if self.config.normalize_advantage:
                adv_std = advantage.std(unbiased=False)
                advantage = (advantage - advantage.mean()) / (adv_std + 1e-8)
            weights = torch.exp(advantage / self.config.awr_temperature)
            if self.config.awr_clip_max is not None:
                weights = torch.clamp(weights, max=self.config.awr_clip_max)
        loss = -(weights.detach() * log_prob).mean()
        info = {
            "advantage_mean": advantage.mean().item(),
            "weight_mean": weights.mean().item(),
        }
        return loss, info

    def compute_loss_actor_bc(
        self,
        observations: dict[str, Tensor],
        actions: Tensor,
    ) -> tuple[Tensor, dict]:
        self._ensure_actor_encoder_trainable()
        dist, _ = self._actor_distribution(
            observations=observations,
            observation_features=None,
            detach_encoder=False,
        )
        log_prob = dist.log_prob(actions)
        mean_log_prob = log_prob.mean()
        loss = -mean_log_prob
        info = {
            "log_prob_mean": mean_log_prob.item(),
        }
        return loss, info

    # --------------------------------------------------------------------- #
    # Utilities
    # --------------------------------------------------------------------- #
    def update_target_networks(self) -> None:
        tau = self.config.critic_target_update_weight
        for target_param, param in zip(self.critic_target.parameters(), self.critic_ensemble.parameters(), strict=True):
            target_param.data.lerp_(param.data, tau)

        value_tau = self.config.value_target_update_weight
        for target_param, param in zip(self.value_target_head.parameters(), self.value_head.parameters(), strict=True):
            target_param.data.lerp_(param.data, value_tau)

        if self.config.num_discrete_actions is not None:
            discrete_tau = self.config.critic_target_update_weight
            for target_param, param in zip(
                self.discrete_critic_target.parameters(), self.discrete_critic.parameters(), strict=True
            ):
                target_param.data.lerp_(param.data, discrete_tau)

    def _actor_distribution(
        self,
        observations: dict[str, Tensor],
        observation_features: Tensor | None = None,
        detach_encoder: bool | None = None,
    ) -> tuple[TanhMultivariateNormalDiag, Tensor]:
        obs_enc = self.actor.encoder(
            observations,
            cache=observation_features,
            detach=self.actor.encoder_is_shared if detach_encoder is None else detach_encoder,
        )
        outputs = self.actor.network(obs_enc)
        means = self.actor.mean_layer(outputs)
        if self.actor.fixed_std is None:
            log_std = self.actor.std_layer(outputs)
            std = torch.exp(log_std)
            std = torch.clamp(std, self.actor.std_min, self.actor.std_max)
        else:
            std = self.actor.fixed_std.expand_as(means)

        dist = TanhMultivariateNormalDiag(
            loc=means,
            scale_diag=std,
            low=self.actor.action_low_bound,
            high=self.actor.action_high_bound,
        )
        return dist, means

    def compute_loss_discrete_critic(
        self,
        observations: dict[str, Tensor],
        actions: Tensor,
        rewards: Tensor,
        next_observations: dict[str, Tensor],
        done: Tensor,
        observation_features: Tensor | None = None,
        next_observation_features: Tensor | None = None,
        complementary_info: dict | None = None,
        return_q_info: bool = False,
    ) -> tuple[Tensor, dict] | Tensor:
        """Compute loss for discrete critic using DQN-style Q-learning.

        For offline RL, we use expectile regression similar to IQL for the value function,
        but for discrete actions we use DQN-style Q-learning with target network.
        """
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
            next_discrete_qs, next_discrete_qs_mean, next_discrete_qs_std = self.discrete_critic_forward(
                next_observations, use_target=False, observation_features=next_observation_features
            )
            best_next_discrete_action = torch.argmax(next_discrete_qs, dim=-1, keepdim=True)

            # Get target Q-values from target network
            target_next_discrete_qs, target_next_discrete_qs_mean, target_next_discrete_qs_std = (
                self.discrete_critic_forward(
                    observations=next_observations,
                    use_target=True,
                    observation_features=next_observation_features,
                )
            )

            # Use gather to select Q-values for best actions
            target_next_discrete_q = torch.gather(
                target_next_discrete_qs, dim=1, index=best_next_discrete_action
            ).squeeze(-1)

            # Compute target Q-value with Bellman equation
            rewards_discrete = rewards
            if discrete_penalties is not None:
                rewards_discrete = rewards + discrete_penalties
            target_discrete_q = rewards_discrete + (1 - done.float()) * self.config.discount * target_next_discrete_q

        # Get predicted Q-values for current observations
        predicted_discrete_qs, predicted_discrete_qs_mean, predicted_discrete_qs_std = self.discrete_critic_forward(
            observations=observations, use_target=False, observation_features=observation_features
        )

        # Use gather to select Q-values for taken actions
        predicted_discrete_q = torch.gather(predicted_discrete_qs, dim=1, index=actions_discrete).squeeze(-1)

        # Compute MSE loss between predicted and target Q-values
        discrete_critic_loss = F.mse_loss(input=predicted_discrete_q, target=target_discrete_q)

        if return_q_info:
            q_info = {}
            q_info["discrete_q_preds_mean"] = predicted_discrete_qs_mean.mean().item()
            q_info["discrete_q_preds_std"] = predicted_discrete_qs_std.mean().item()
            q_info["discrete_q_targets_mean"] = target_next_discrete_qs_mean.mean().item()
            q_info["discrete_q_targets_std"] = target_next_discrete_qs_std.mean().item()
            return discrete_critic_loss, q_info
        return discrete_critic_loss

    def _ensure_actor_encoder_trainable(self) -> None:
        for param in self.actor.encoder.parameters():
            if not param.requires_grad:
                print(f"BC Stage: Ensuring encoder {param.name()} is trainable")
                param.requires_grad_(True)
