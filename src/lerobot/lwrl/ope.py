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

from lerobot.rl.buffer import ReplayBuffer


@torch.no_grad()
def amq_score_from_buffer(buf: ReplayBuffer, key: str = "q_min") -> tuple[float, int]:
    """Compute AM-Q score from buffer's complementary_info.

    AM-Q (Average Model Q-value) is computed as the sum of Q-values along
    policy-collected episodes, normalized by the number of episodes.

    Args:
        buf: ReplayBuffer containing transitions with complementary_info
        key: Key in complementary_info containing Q-values (default: "q_min")

    Returns:
        Tuple of (AM-Q score, number of episodes). Returns (-inf, 0) if no data.
    """
    if not hasattr(buf, "initialized") or not buf.initialized:
        return float("-inf"), 0

    if not hasattr(buf, "has_complementary_info") or not buf.has_complementary_info:
        return float("-inf"), 0

    qvals: Tensor | None = buf.complementary_info.get(key, None)
    if qvals is None or qvals.numel() == 0:
        return float("-inf"), 0

    # Sum across all transitions; normalize by number of episodes for scale invariance
    # If the actor logged 'episode_id', we can count distinct episodes
    ep_ids = buf.complementary_info.get("episode_id", None)
    if ep_ids is None:
        eps = 1
    else:
        eps = int(torch.unique(ep_ids).numel())
        eps = max(eps, 1)

    # AM-Q = sum of Q-values normalized by episodes
    score = float(qvals.sum().item()) / eps
    return score, eps

