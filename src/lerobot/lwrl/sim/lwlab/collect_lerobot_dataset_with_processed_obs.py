#!/usr/bin/env python3
"""
LeRobot Dataset Collection Script
Collect data from real environments and convert to LeRobot dataset format

Main features:
1. Interact with real environment and collect data to buffer
2. Convert buffer data to LeRobot dataset
3. Support flexible parameter configuration
"""

import torch
import argparse
import os
import shutil
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import copy
from tqdm import tqdm

# 导入必要的模块
from lwlab.distributed.proxy import RemoteEnv
from lwlab.utils.config_loader import config_loader
from policy.maniskill_ppo.agent import PPOArgs, PPO
from policy.maniskill_ppo.agent import observation as process_maniskill_ppo_observation
from lerobot.lwrl.buffer_batched import ParallelReplayBuffer, BatchTransition
from lerobot.utils.transition import move_transition_to_device

# hilserl processors and utils
from lerobot.rl.gym_manipulator import (
    create_transition,
    # make_processors,
    # make_robot_env,
    # step_env_and_process_transition,
)
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.processor import TransitionKey
from lerobot.configs import parser

def process_maniskill_ppo_observation_override(obs):
    """
    Override process maniskill ppo observation
    Args:
        obs: dict
    """
    # pop "hand camera" image
    obs.pop("image_hand") if "image_hand" in obs else None
    return process_maniskill_ppo_observation(obs)

def save_image_uint8(image_tensor, filepath):
    """
    save uint8 image to file
    Args:
        image_tensor: torch.Tensor or numpy.ndarray
        filepath: str
    """
    import numpy as np
    from PIL import Image
    import os

    # 确保输入是numpy数组格式
    if isinstance(image_tensor, torch.Tensor):
        # 将张量转换为numpy数组
        image_np = image_tensor.detach().cpu().numpy()
    else:
        image_np = image_tensor
    
    # ensure data type is uint8
    if image_np.dtype != np.uint8:
        image_np = (image_np*255).astype(np.uint8)
    
    # ensure shape is [H, W, 3]
    if len(image_np.shape) != 3:
        raise ValueError(f"image should has 3 dimensions, but got {image_np.shape}")
    elif image_np.shape[2] != 3:
        import einops
        image_np = einops.rearrange(image_np, "c h w -> h w c")

    pil_image = Image.fromarray(image_np, 'RGB')
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    pil_image.save(filepath)
    print(f"image saved to: {filepath}")


@dataclass
class CollectionArgs:
    """Data collection parameter configuration"""
    # Environment configuration
    task_config: str = "lerobot_liftobj_visual_hilserl_play"
    env_address: str = "0.0.0.0"
    env_port: int = 50000
    env_authkey: str = "lightwheel"
    
    # Data collection configuration
    num_steps: int = 1000
    device: str = "cuda:0"
    storage_device: str = "cpu"
    
    # Model configuration
    checkpoint: Optional[str] = None
    deterministic: bool = False
    
    # Dataset configuration
    repo_id: str = "collected_dataset"
    task_name: str = "data_collection"
    fps: int = 20
    root_dir: str = "./datasets"
    
    # PPO configuration
    ppo: PPOArgs = field(default_factory=PPOArgs)


class DataCollector:
    """Data collector class"""
    
    def __init__(self, args: CollectionArgs, cfg:TrainRLServerPipelineConfig):
        self.args = args
        self.env = None
        self.agent = None
        self.buffer = None
        self.cfg = cfg
        
    def setup_environment(self):
        """Setup environment"""
        print("Setting up environment...")
        from lerobot.lwrl.sim.lwlab.env_lwlab import make_lwlab_robot_env, make_lwlab_processors
        self.env, teleop_device = make_lwlab_robot_env(cfg=self.cfg.env)
        print(f"Environment setup complete, parallel environments: {self.env.num_envs}")
        print(f"Setting up processors...")
        self.env_processor, self.action_processor = make_lwlab_processors(self.env, teleop_device, self.cfg.env, self.cfg.policy.device)
        print(f"Processors setup complete")
        
    def setup_agent(self):
        """Setup agent"""
        print("Setting up agent...")
        obs, _ = self.env.reset()
        obs = obs
        
        self.agent = PPO(
            self.env, 
            process_maniskill_ppo_observation_override(copy.deepcopy(
                obs['policy'] if 'policy' in obs else obs)), 
            self.args.ppo, 
            self.args.device, 
            train=False
        )
        
        assert self.args.checkpoint is not None, "Checkpoint is required"
        print(f"Loading checkpoint: {self.args.checkpoint}")
        self.agent.load_model(self.args.checkpoint)            
        print("Agent setup complete")
        
    def setup_buffer(self):
        """Setup buffer"""
        print("Setting up buffer...")
        self.buffer = ParallelReplayBuffer(
            capacity=self.args.num_steps * self.env.num_envs * 2,  # Extra capacity for safety
            num_envs=self.env.num_envs,
            device=self.args.device,
            storage_device=self.args.storage_device
        )
        print(f"Buffer setup complete, capacity: {self.args.num_steps * 2}")
        
    def collect_data(self):
        """Collect data"""
        print(f"Starting data collection: {self.args.num_steps} steps, {self.env.num_envs} parallel environments")
        
        # Reset environment
        raw_obs, _ = self.env.reset()

        # processor
        obs, info = self.env.reset()
        self.env_processor.reset()
        self.action_processor.reset()

        # Process initial observation
        transition = create_transition(
            observation=obs,
            reward=torch.zeros((self.env.num_envs,), dtype=torch.float32, device=self.env.device),
            done=torch.zeros((self.env.num_envs,), dtype=torch.bool, device=self.env.device),
            truncated=torch.zeros((self.env.num_envs,), dtype=torch.bool, device=self.env.device),
            info=info)
        transition = self.env_processor(transition)
        
        step_count = 0
        success_count = 0
        episode_count = 0
        
        # Create progress bar
        pbar = tqdm(total=self.args.num_steps, desc="Data collection progress")
        
        with torch.inference_mode():
            while step_count < self.args.num_steps:
                # Get actions
                actions = self.agent.agent.get_action(
                    process_maniskill_ppo_observation_override(copy.deepcopy(
                        raw_obs['policy'] if 'policy' in raw_obs else raw_obs)), 
                    deterministic=self.args.deterministic
                )

                # process
                observation = {
                    k: v for k, v in transition[TransitionKey.OBSERVATION].items() 
                }

                #! copied from step_lwlab_env_and_process_transition
                # needed to copy function here because we need to 
                # 1) step raw action
                # 2) get raw next observation (for baseline rl policy)
            
                # from lerobot.lwrl.sim.lwlab.env_lwlab import step_lwlab_env_and_process_transition
                transition[TransitionKey.ACTION] = actions
                transition[TransitionKey.OBSERVATION] = (
                    raw_obs if hasattr(self.env, "get_raw_joint_positions") else {}
                )
                processed_action_transition = self.action_processor(transition)
                processed_action = processed_action_transition[TransitionKey.ACTION]

                # core step function, need to step raw action
                # obs, reward, terminated, truncated, info = self.env.step(processed_action)
                next_raw_obs, reward, terminated, truncated, info = self.env.step(actions)
                # import ipdb; ipdb.set_trace()

                # Statistics
                success_count += info['is_success'].sum().item()
                episode_count += (terminated | truncated).sum().item()

                reward = reward + processed_action_transition[TransitionKey.REWARD]
                #! changed to batched or
                terminated = torch.logical_or(terminated, processed_action_transition[TransitionKey.DONE])
                truncated = torch.logical_or(truncated, processed_action_transition[TransitionKey.TRUNCATED])
                complementary_data = processed_action_transition[TransitionKey.COMPLEMENTARY_DATA].copy()
                new_info = processed_action_transition[TransitionKey.INFO].copy()
                new_info.update(info)

                new_transition = create_transition(
                    observation=next_raw_obs, # use raw obs
                    action=processed_action,
                    reward=reward,
                    done=terminated,
                    truncated=truncated,
                    info=new_info,
                    complementary_data=complementary_data,
                )
                new_transition = self.env_processor(new_transition)


                # Extract values from processed transition
                next_observation = {
                    k: v
                    for k, v in new_transition[TransitionKey.OBSERVATION].items()
                }

                executed_action = new_transition[TransitionKey.ACTION]
                reward = new_transition[TransitionKey.REWARD]
                done = new_transition.get(TransitionKey.DONE, torch.tensor(False))
                truncated = new_transition.get(TransitionKey.TRUNCATED, torch.tensor(False))
                info = new_transition.get(TransitionKey.INFO, {})

                if torch.any(done) or torch.any(truncated):
                    new_transition_with_reset = new_transition
                    # re-write done and truncated
                    new_transition_with_reset[TransitionKey.DONE] = torch.zeros_like(done, device=self.args.device, dtype=torch.bool)
                    new_transition_with_reset[TransitionKey.TRUNCATED] = torch.zeros_like(truncated, device=self.args.device, dtype=torch.bool)
                    new_transition_with_reset[TransitionKey.REWARD] = torch.zeros_like(reward, device=self.args.device, dtype=torch.float32)
                    new_transition_with_reset[TransitionKey.INFO] = {}
                    #! original code will reset processor here, but skip for now
                    # TODO: need to implement reset processor per env index
                    # env_processor.reset()
                    # action_processor.reset()
                    
                    # recreate real transition and overwrite next observation (pass processer)
                    next_observation_raw = info['final_obs'] # replace with last obs before reset
                    new_transition_raw = create_transition(
                        observation=next_observation_raw, info=info,
                        done=torch.zeros_like(done, device=self.args.device, dtype=torch.bool),
                        truncated=torch.zeros_like(truncated, device=self.args.device, dtype=torch.bool),
                        reward=torch.zeros_like(reward, device=self.args.device, dtype=torch.float32),
                    )
                    # Extract values from processed transition
                    new_transition = self.env_processor(new_transition_raw)
                    next_observation = {
                        k: v
                        for k, v in new_transition[TransitionKey.OBSERVATION].items()
                    }
                    # make sure those will not be used!! (only create to use processer)
                    del new_transition, new_transition_raw

                    info.pop('final_obs') # remove final_obs from info to save space
                
                # Create transition data
                parallel_transition = BatchTransition(
                    state=copy.deepcopy(observation),
                    action=executed_action,
                    reward=reward,
                    next_state=copy.deepcopy(next_observation),
                    done=done,
                    truncated=truncated,
                    complementary_info={
                        "is_success": info.get('is_success', torch.zeros_like(done, device=self.args.device)).to(torch.float32),
                        "ee_pose": info.get('ee_pose', torch.zeros((self.env.num_envs, 7), device=self.args.device)),
                    },
                )

                # # test visualize image
                # save_image_uint8((parallel_transition['state']['observation.images.image_global'][0], './buffer1.png')
                # self.buffer.states['observation.images.image_global'][0, 0]
                
                # assign obs to the next obs and continue the rollout
                if torch.any(done) or torch.any(truncated):
                    transition = new_transition_with_reset
                else:
                    transition = new_transition
                
                raw_obs = next_raw_obs
                step_count += 1
                
                # Update progress bar
                pbar.update(1)
                
                # Print statistics periodically
                if step_count % 100 == 0:
                    pbar.set_postfix({
                        'success': success_count,
                        'episodes': episode_count,
                        'buffer_size': len(self.buffer)
                    })

                # Move to storage device
                tr = move_transition_to_device(parallel_transition, device=self.buffer.storage_device)
                
                # Add to buffer
                self.buffer.add(
                    **tr
                )
                
        
        pbar.close()
        
        print(f"\nData collection complete!")
        print(f"Total steps: {step_count}")
        print(f"Success count: {success_count}")
        print(f"Episode count: {episode_count}")
        print(f"Buffer size: {len(self.buffer)}")
        
        return self.buffer
        
    def save_dataset(self):
        """Save dataset"""
        print("Saving dataset...")
        
        # Ensure root directory exists
        root_path = Path(self.args.root_dir)
        # make parent path if not exists
        root_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if there is success data
        if "is_success" in self.buffer.complementary_info:
            is_success = self.buffer.complementary_info["is_success"].sum() > 0
            if not is_success:
                print("Warning: No success data in buffer")
        
        # Convert to LeRobot dataset
        dataset = self.buffer.to_lerobot_dataset(
            repo_id=self.args.repo_id,
            fps=self.args.fps,
            root=str(root_path),
            task_name=self.args.task_name
        )
        
        print(f"Dataset saved to: {root_path}")
        print(f"Dataset frames: {len(dataset)}")

        # test data
        print("Testing dataset...")
        for i in range(len(dataset)):
            try:
                dataset[i]
            except Exception as e:
                import ipdb; ipdb.set_trace()
                print(f"Error at index {i}, need to re-generate dataset")
        
        # Validate dataset structure
        if len(dataset) > 0:
            sample = dataset[0]
            required_keys = ["action", "next.reward", "next.done"]
            for key in required_keys:
                if key not in sample:
                    print(f"Warning: Missing required key {key}")
            
            print("✓ Dataset structure validation passed")
        else:
            print("Warning: Dataset is empty")
            
        return dataset
        
    def run_collection(self):
        """Run complete data collection workflow"""
        print("=" * 60)
        print("Starting LeRobot Dataset Collection")
        print("=" * 60)
        
        try:
            # Setup components
            self.setup_environment()
            self.setup_agent()
            self.setup_buffer()
            
            # Collect data
            buffer = self.collect_data()
            
            # Save dataset
            dataset = self.save_dataset()
            
            print("\n" + "=" * 60)
            print("🎉 Data collection complete!")
            print("=" * 60)
            
            return dataset
            
        except Exception as e:
            print(f"\n❌ Data collection failed: {e}")
            import traceback
            traceback.print_exc()
            raise


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="LeRobot Dataset Collection Script")
    
    # Environment configuration
    parser.add_argument("--task_config", type=str, 
                       default="lerobot_liftobj_visual_hilserl_play",
                       help="Task configuration file")
    
    # Data collection configuration
    parser.add_argument("--num_steps", type=int, default=100,
                       help="Number of steps to collect")
 
    # Model configuration
    parser.add_argument("--deterministic", action="store_true",
                       help="Use deterministic policy")
    
    # Dataset configuration
    parser.add_argument("--repo_id", type=str, default="rl-autonomy/lerobot-pickup-visual",
                       help="Dataset repository ID")
    parser.add_argument("--task_name", type=str, default="LerobotPickupVisual",
                       help="Task name")
    parser.add_argument("--fps", type=int, default=20,
                       help="Dataset FPS")

    import datetime
    current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    parser.add_argument("--root_dir", type=str, default=f"./data/{current_time}",
                       help="Dataset root directory")
    
    # hilserl args (for obs processing)
    parser.add_argument("--config_path", type=str, default="/home/johndoe/Documents/lerobot-hilserl/src/lerobot/configs/rl/hilserl_sim_lwlab_lerobot/train_lwlab_hil_lerobotPnP_w_data.json",
                       help="Hilserl arguments", required=True)
    
    return parser.parse_args()


@parser.wrap()
def parse_hilserl_args(cfg: TrainRLServerPipelineConfig):
    cfg.validate()
    return cfg

def main(cfg: TrainRLServerPipelineConfig):
    """Main function"""
    # Set multiprocessing start method
    from torch import multiprocessing as mp
    mp.set_start_method("fork", force=True)

    # Parse command line arguments
    args_cli = parse_arguments()
    
    # Load YAML configuration
    yaml_args = config_loader.load(args_cli.task_config)
    args_cli.__dict__.update(yaml_args.__dict__)
    
    # Create collection arguments
    collection_args = CollectionArgs(
        task_config=args_cli.task_config,
        num_steps=args_cli.num_steps,
        checkpoint=args_cli.checkpoint,
        repo_id=args_cli.repo_id,
        task_name=args_cli.task_name,
        fps=args_cli.fps,
        root_dir=args_cli.root_dir,
    )
    
    # Create data collector and run
    collector = DataCollector(collection_args, cfg)
    dataset = collector.run_collection()
    
    print(f"\nDataset saved to: {collection_args.root_dir}")
    print(f"Dataset ID: {collection_args.repo_id}")
    print(f"Task name: {collection_args.task_name}")


if __name__ == "__main__":
    # need to import here for cfg decoding
    from lerobot.policies.sac.modeling_sac import SACPolicy
    cfg = parse_hilserl_args()

    main(cfg)

"""
hilserl
python /home/johndoe/Documents/lerobot-hilserl/src/lerobot/lwrl/sim/lwlab/collect_lerobot_dataset_with_processed_obs.py \
    --config_path="/home/johndoe/Documents/lerobot-hilserl/src/lerobot/configs/rl/hilserl_sim_lwlab_lerobot/train_lwlab_hil_lerobotPnP.json"

"""