import logging
import time
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from lerobot.envs.configs import HILSerlRobotEnvConfig
from lerobot.processor.pipeline import DataProcessorPipeline
from lerobot.processor.core import EnvTransition
from lerobot.processor.converters import identity_transition, create_transition
from lerobot.teleoperators import Teleoperator
from lerobot.processor.core import TransitionKey



def make_lwlab_robot_env(cfg: HILSerlRobotEnvConfig) -> tuple[gym.Env, Any]:
    """Create robot environment from configuration.

    Args:
        cfg: Environment configuration.

    Returns:
        Tuple of (gym environment, teleoperator device).
    """
    # Check if this is a GymHIL simulation environment
    assert cfg.type == "lwlab", "LwLab environment must be provided"

    from lwlab.distributed.proxy import RemoteEnv
    from torch import multiprocessing as mp
    mp.set_start_method("fork", force=True)

    env = RemoteEnv.make(address=(cfg.address, cfg.port), authkey=bytes(cfg.authkey, 'utf-8'))
    env = env.unwrapped
    env.reset()

    return env, None


def make_lwlab_processors(
    env: gym.Env, teleop_device: Teleoperator | None, cfg: HILSerlRobotEnvConfig, device: str = "cpu"
) -> tuple[
    DataProcessorPipeline[EnvTransition, EnvTransition], DataProcessorPipeline[EnvTransition, EnvTransition]
]:
    """Create environment and action processors.

    Args:
        env: Robot environment instance.
        teleop_device: Teleoperator device for intervention.
        cfg: Processor configuration.
        device: Target device for computations.

    Returns:
        Tuple of (environment processor, action processor).
    """

    assert cfg.type == "lwlab", "LwLab environment must be provided"

    from lerobot.lwrl.sim.lwlab.lwlab_processor_steps import (
        RepackLwlabObservationProcessorStep,
        LwlabSparseRewardProcessorStep,
        LwlabNumpy2TorchActionProcessorStep,
    )

    # no need to process action here
    action_pipeline_steps = [
        LwlabNumpy2TorchActionProcessorStep(device=env.device),
    ]
    env_pipeline_steps = [
        RepackLwlabObservationProcessorStep(
            ENV_STATE_KEYS=cfg.processor.ENV_STATE_KEYS, # privileged keys
            OBS_STATE_KEYS=cfg.processor.OBS_STATE_KEYS, # observation keys,
            CAMERA_KEYS=cfg.processor.CAMERA_KEYS, # camera keys
            features=cfg.features
        ),
        LwlabSparseRewardProcessorStep(),
    ]
    
    return DataProcessorPipeline(
        steps=env_pipeline_steps, to_transition=identity_transition, to_output=identity_transition
    ), DataProcessorPipeline(
        steps=action_pipeline_steps, to_transition=identity_transition, to_output=identity_transition
    )


def step_lwlab_env_and_process_transition(
    env: gym.Env,
    transition: EnvTransition,
    action: torch.Tensor,
    env_processor: DataProcessorPipeline[EnvTransition, EnvTransition],
    action_processor: DataProcessorPipeline[EnvTransition, EnvTransition],
) -> EnvTransition:
    """
    Execute one step with processor pipeline.

    Args:
        env: The robot environment
        transition: Current transition state
        action: Action to execute
        env_processor: Environment processor
        action_processor: Action processor

    Returns:
        Processed transition with updated state.
    """

    # Create action transition
    transition[TransitionKey.ACTION] = action
    transition[TransitionKey.OBSERVATION] = (
        env.get_raw_joint_positions() if hasattr(env, "get_raw_joint_positions") else {}
    )
    processed_action_transition = action_processor(transition)
    processed_action = processed_action_transition[TransitionKey.ACTION]

    obs, reward, terminated, truncated, info = env.step(processed_action)

    target_device = processed_action_transition[TransitionKey.DONE].device
    terminated = terminated.to(target_device)
    truncated = truncated.to(target_device)
    reward = reward.to(processed_action_transition[TransitionKey.REWARD].device)

    reward = reward + processed_action_transition[TransitionKey.REWARD]
    #! changed to batched or
    terminated = torch.logical_or(terminated, processed_action_transition[TransitionKey.DONE])
    truncated = torch.logical_or(truncated, processed_action_transition[TransitionKey.TRUNCATED])
    complementary_data = processed_action_transition[TransitionKey.COMPLEMENTARY_DATA].copy()
    new_info = processed_action_transition[TransitionKey.INFO].copy()
    new_info.update(info)

    new_transition = create_transition(
        observation=obs,
        action=processed_action,
        reward=reward,
        done=terminated,
        truncated=truncated,
        info=new_info,
        complementary_data=complementary_data,
    )
    new_transition = env_processor(new_transition)

    return new_transition