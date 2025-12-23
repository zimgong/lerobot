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

from dataclasses import dataclass
from typing import Any

import torch

from lerobot.policies.sac.configuration_sac_go1 import SACGO1Config

from go1.internvl.train.dataset import build_transform
from go1.lerobot.dataset_lerobot import WrappedLeRobotDataset, tensor_to_pil
from go1.lerobot.dataset_transforms import make_conversation


@dataclass
class GO1ObservationPreprocessor:
    """Prepare GO-1 multimodal inputs from raw observations."""

    config: SACGO1Config
    tokenizer: Any
    vla_model: Any

    def __post_init__(self) -> None:
        self._camera_alias_to_source_key = {
            alias: self.config.space_repack[alias]
            for alias in ("cam_head_color", "cam_hand_right_color", "cam_hand_left_color")
            if alias in self.config.space_repack
        }
        self._prompt_source_key = self.config.space_repack.get("final_prompt")
        self._default_prompt = self.config.default_prompt
        self._image_transform = build_transform(
            is_train=True,
            input_size=self.vla_model.config.force_image_size,
            pad2square=self.vla_model.config.pad2square,
            normalize_type="imagenet",
        )

    def __call__(self, observations: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.prepare_batch(observations)

    def prepare_batch(self, observations: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Convert a batch of raw observations into GO-1 multimodal inputs."""
        if not observations:
            raise ValueError("Observations dictionary is empty.")

        batch_size = next(iter(observations.values())).shape[0]

        # Process batch items in parallel
        features = []
        for b in range(batch_size):
            features.append(self._prepare_single({k: v[b : b + 1] for k, v in observations.items()}))

        observations["pixel_values"] = torch.cat([ft["pixel_values"] for ft in features], dim=0).to(device=self.vla_model.device)
        observations["input_ids"] = torch.stack([ft["input_ids"] for ft in features], dim=0).to(device=self.vla_model.device)
        observations["attention_mask"] = torch.stack([ft["attention_mask"] for ft in features], dim=0).to(device=self.vla_model.device)
        observations["position_ids"] = torch.stack([ft["position_ids"] for ft in features], dim=0).to(device=self.vla_model.device)
        observations["image_flags"] = torch.cat([ft["image_flags"] for ft in features], dim=0).to(device=self.vla_model.device)
        observations["labels"] = torch.stack([ft["labels"] for ft in features], dim=0).to(device=self.vla_model.device)

        return observations

    def _prepare_single(self, observation: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # Convert observation tensors to a format expected by WrappedLeRobotDataset
        raw_target: dict[str, Any] = {}
        for alias, source_key in self._camera_alias_to_source_key.items():
            if source_key not in observation:
                continue
            camera_tensor = observation[source_key][0].permute(1, 2, 0)
            raw_target[alias] = tensor_to_pil(camera_tensor)

        if self._prompt_source_key and self._prompt_source_key in observation:
            prompt = observation[self._prompt_source_key]
        else:
            prompt = self._default_prompt
        raw_target["final_prompt"] = make_conversation(prompt=prompt)

        features = WrappedLeRobotDataset.multi_image_get_item(
            raw_target=raw_target,
            img_transform=self._image_transform,
            text_tokenizer=self.tokenizer,
            num_image_token=self.vla_model.num_image_token,
            use_thumbnail=self.vla_model.config.use_thumbnail,
            min_dynamic_patch=self.vla_model.config.min_dynamic_patch,
            max_dynamic_patch=self.vla_model.config.max_dynamic_patch,
            image_size=self.vla_model.config.force_image_size,
        )

        return features
