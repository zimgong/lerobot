# !/usr/bin/env python

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

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.optim.optimizers import MultiAdamConfig
from lerobot.policies.sac.configuration_sac import SACConfig


@PreTrainedConfig.register_subclass("offline")
@dataclass
class OfflineIQLConfig(SACConfig):
    """Configuration for the Offline (IQL/AWR) policy.

    This configuration reuses the SAC wiring (encoders, actor, critics) while exposing
    the hyperparameters required by the Implicit Q-Learning losses used during the
    offline stage (expectile regression for the value network and advantage weighted
    regression for the actor).
    """

    # Offline-specific critic/actor hyperparameters
    expectile_tau: float = 0.7
    awr_temperature: float = 3.0
    awr_clip_max: float | None = 100.0
    normalize_advantage: bool = True

    # PPO offline (optional)
    ppo_clip_eps: float = 0.2
    # "awr" (default) or "ppo"
    actor_update: str = "awr"

    # Optimizer hyperparameters
    value_lr: float = 3e-4

    # Polyak averaging coefficient for both critics and values
    value_target_update_weight: float | None = None

    # Offline training loop control
    offline_epochs: int = 10

    def __post_init__(self):
        super().__post_init__()
        if self.value_target_update_weight is None:
            # Default to critic target update if not specified
            self.value_target_update_weight = self.critic_target_update_weight

    def get_optimizer_preset(self) -> MultiAdamConfig:
        # NOTE: We keep temperature off (no entropy tuning in offline stage)
        optimizer_groups = {
            "actor": {"lr": self.actor_lr},
            "critic": {"lr": self.critic_lr},
            "value": {"lr": self.value_lr},
        }
        if self.num_discrete_actions is not None:
            optimizer_groups["discrete_critic"] = {"lr": self.critic_lr}
        return MultiAdamConfig(
            weight_decay=0.0,
            optimizer_groups=optimizer_groups,
        )
