# !/usr/bin/env python

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
import logging

from lerobot.cameras import opencv  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy
from lerobot.processor import TransitionKey
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    make_robot_from_config,
    so100_follower,
)
from lerobot.teleoperators import (
    gamepad,  # noqa: F401
    so101_leader,  # noqa: F401
)

from .gym_manipulator import (
    create_transition,
    make_processors,
    make_robot_env,
    step_env_and_process_transition,
)

logging.basicConfig(level=logging.INFO)


@parser.wrap()
def main(cfg: TrainRLServerPipelineConfig):
    env_cfg = cfg.env
    env, teleop_device = make_robot_env(env_cfg)

    policy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
    )
    pretrained_policy_name_or_path = "/home/zimu.gong/huggingface/lerobot/outputs/train/2025-10-27/11-48-25_default_4090/checkpoints/last/pretrained_model"
    policy.from_pretrained(pretrained_policy_name_or_path)
    policy.eval()

    env_processor, action_processor = make_processors(env, teleop_device, cfg.env, cfg.policy.device)
    sum_reward_episode = []

    n_episodes = 10

    for _ in range(n_episodes):
        obs, info = env.reset()
        env_processor.reset()
        action_processor.reset()
        transition = create_transition(observation=obs, info=info)
        transition = env_processor(transition)
        episode_reward = 0.0
        while True:
            observation = {
                k: v for k, v in transition[TransitionKey.OBSERVATION].items() if k in cfg.policy.input_features
            }
            action = policy.select_action(observation)
            new_transition = step_env_and_process_transition(
                env=env,
                transition=transition,
                action=action,
                env_processor=env_processor,
                action_processor=action_processor,
            )

            reward = new_transition[TransitionKey.REWARD]
            done = new_transition.get(TransitionKey.DONE, False)
            truncated = new_transition.get(TransitionKey.TRUNCATED, False)

            transition = new_transition

            episode_reward += reward
            if done or truncated:
                break
        sum_reward_episode.append(episode_reward)

    logging.info(f"Success after 20 steps {sum_reward_episode}")
    logging.info(f"success rate {sum(sum_reward_episode) / len(sum_reward_episode)}")


if __name__ == "__main__":
    main()
