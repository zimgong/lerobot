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

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import MultiAdamConfig
from lerobot.policies.sac.configuration_sac import (
    is_image_feature,
    ConcurrencyConfig,
    ActorLearnerConfig,
    CriticNetworkConfig,
    ActorNetworkConfig,
    PolicyConfig,
)
from lerobot.utils.constants import ACTION, OBS_IMAGE, OBS_STATE

from go1.configs.go1_base_cfg import BaseModelArguments
from go1.tools.env_parse import get_bool_env

DEBUG_MODE = get_bool_env("DEBUG_MODE")


@dataclass
class GOModelArguments(BaseModelArguments):
    model_name_or_path: str = field(default="agibot-world/GO-1-Air")
    freeze_llm: bool = field(default=False if not DEBUG_MODE else True)
    freeze_backbone: bool = field(default=False if not DEBUG_MODE else True)
    freeze_mlp: bool = field(default=False if not DEBUG_MODE else True)
    action_chunk_size: int = field(default=10)
    latent_planning: bool = field(default=False)    


@PreTrainedConfig.register_subclass("sac_go1")
@dataclass
class SACGO1Config(PreTrainedConfig):
    """Soft Actor-Critic (SAC) with AgiBot GO-1 configuration."""

    # Mapping of feature types to normalization modes
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ENV": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # Statistics for normalizing different types of inputs
    dataset_stats: dict[str, dict[str, list[float]]] | None = field(
        default_factory=lambda: {
            OBS_IMAGE: {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            OBS_STATE: {
                "min": [0.0, 0.0],
                "max": [1.0, 1.0],
            },
            ACTION: {
                "min": [0.0, 0.0, 0.0],
                "max": [1.0, 1.0, 1.0],
            },
        }
    )

    # Arguments for obs / action space
    space_repack: dict = field(
        default_factory=lambda: {
            "state": "state",
            "action": "action",
            "cam_head_color": "cam_head_color",
            "cam_hand_left_color": "cam_hand_left_color",
            "cam_hand_right_color": "cam_hand_right_color",
            "final_prompt": "final_prompt",
        }
    )
    ctrl_freq: int = field(default=30)
    default_prompt: str = field(default="your instruction here")

    # Architecture specifics
    # Device to run the model on (e.g., "cuda", "cpu")
    device: str = "cpu"
    # Device to store the model on
    storage_device: str = "cpu"
    # Name of the vision encoder model (Set to "helper2424/resnet10" for hil serl resnet10)
    vision_encoder_name: str | None = None
    # Whether to freeze the vision encoder during training
    freeze_vision_encoder: bool = True
    # Whether to use a shared encoder for actor and critic
    shared_encoder: bool = True
    # Number of discrete actions, eg for gripper actions
    num_discrete_actions: int | None = None
    # Dimension of the image embedding pooling
    image_embedding_pooling_dim: int = 8

    # Training parameter
    # Number of steps for online training
    online_steps: int = 1000000
    # Capacity of the online replay buffer
    online_buffer_capacity: int = 100000
    # Capacity of the offline replay buffer
    offline_buffer_capacity: int = 100000
    # Whether to use asynchronous prefetching for the buffers
    async_prefetch: bool = False
    # Number of steps before learning starts
    online_step_before_learning: int = 100
    # Frequency of policy updates
    policy_update_freq: int = 1
    # Number of steps before actor learning starts (learn critic only before for stability)
    critic_step_before_actor: int = 0

    # SAC algorithm parameters
    # Discount factor for the SAC algorithm
    discount: float = 0.99
    # Initial temperature value
    temperature_init: float = 1.0
    # Number of critics in the ensemble
    num_critics: int = 2
    # Number of subsampled critics for training
    num_subsample_critics: int | None = None
    # Learning rate for the critic network
    critic_lr: float = 3e-4
    # Learning rate for the actor network
    actor_lr: float = 3e-4
    # Learning rate for the temperature parameter
    temperature_lr: float = 3e-4
    # Weight for the critic target update
    critic_target_update_weight: float = 0.005
    # Update-to-data ratio for the UTD algorithm (If you want enable utd_ratio, you need to set it to >1)
    utd_ratio: int = 1
    # Hidden dimension size for the state encoder
    state_encoder_hidden_dim: int = 256
    # Dimension of the latent space
    latent_dim: int = 256
    # Target entropy for the SAC algorithm
    target_entropy: float | None = None
    # Whether to use backup entropy for the SAC algorithm
    use_backup_entropy: bool = True
    # Gradient clipping norm for the SAC algorithm
    grad_clip_norm: float = 40.0

    # Network configuration
    # Configuration for the GO-1 model
    go1_model_kwargs: GOModelArguments = field(default_factory=GOModelArguments)
    # Configuration for the critic network architecture
    critic_network_kwargs: CriticNetworkConfig = field(default_factory=CriticNetworkConfig)
    # Configuration for the actor network architecture
    actor_network_kwargs: ActorNetworkConfig = field(default_factory=ActorNetworkConfig)
    # Configuration for the policy parameters
    policy_kwargs: PolicyConfig = field(default_factory=PolicyConfig)
    # Configuration for the discrete critic network
    discrete_critic_network_kwargs: CriticNetworkConfig = field(default_factory=CriticNetworkConfig)
    # Configuration for actor-learner architecture
    actor_learner_config: ActorLearnerConfig = field(default_factory=ActorLearnerConfig)
    # Configuration for concurrency settings (you can use threads or processes for the actor and learner)
    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)

    # Optimizations
    use_torch_compile: bool = True

    def __post_init__(self):
        super().__post_init__()
        # Any validation specific to SAC configuration

    def get_optimizer_preset(self) -> MultiAdamConfig:
        return MultiAdamConfig(
            weight_decay=0.0,
            optimizer_groups={
                "actor": {"lr": self.actor_lr},
                "critic": {"lr": self.critic_lr},
                "temperature": {"lr": self.temperature_lr},
            },
        )

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        has_image = any(is_image_feature(key) for key in self.input_features)
        has_state = OBS_STATE in self.input_features

        if not (has_state or has_image):
            raise ValueError(
                "You must provide either 'observation.state' or an image observation (key starting with 'observation.image') in the input features"
            )

        if ACTION not in self.output_features:
            raise ValueError("You must provide 'action' in the output features")

    @property
    def image_features(self) -> list[str]:
        return [key for key in self.input_features if is_image_feature(key)]

    @property
    def observation_delta_indices(self) -> list:
        return None

    @property
    def action_delta_indices(self) -> list:
        return None  # SAC typically predicts one action at a time

    @property
    def reward_delta_indices(self) -> None:
        return None
