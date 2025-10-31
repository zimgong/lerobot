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

def process_maniskill_ppo_observation_override(obs):
    """
    Override process maniskill ppo observation
    Args:
        obs: dict
    """
    # pop "hand camera" image
    obs.pop("image_hand") if "image_hand" in obs else None
    obs.pop("ee_pose") if "ee_pose" in obs else None
    return process_maniskill_ppo_observation(obs)

def _convert_images_to_uint8_chw(obs_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Convert image tensors in an observation dict to uint8 and CHW/ BCHW.

    - Keeps keys unchanged
    - Handles both 3D (H, W, C) and 4D (B, H, W, C) tensors
    - Only transforms tensors that look like images (last dim 1 or 3)
    """
    if obs_dict is None:
        return obs_dict

    processed = {}
    for key, value in obs_dict.items():
        v = value
        try:
            if isinstance(v, torch.Tensor) and v.ndim in (3, 4):
                if v.shape[-1] in (1, 3) and v[1:].numel() > 64**2:
                    # Move channel-last to channel-first
                    if v.ndim == 3:  # H, W, C -> C, H, W
                        v = v.permute(2, 0, 1).contiguous()
                    else:  # B, H, W, C -> B, C, H, W
                        v = v.permute(0, 3, 1, 2).contiguous()
                    # Cast to uint8 as requested
                    if v.dtype != torch.uint8:
                        v = v.to(torch.uint8)
                    
                    # uint8 to float32
                    v = v.to(torch.float32)
                    v /= 255.0
        except Exception:
            # If any unexpected structure, leave as-is
            pass
        processed[key] = v
    return processed

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
    root_dir: str = "./data"
    
    # PPO configuration
    ppo: PPOArgs = field(default_factory=PPOArgs)


class DataCollector:
    """Data collector class"""
    
    def __init__(self, args: CollectionArgs):
        self.args = args
        self.env = None
        self.agent = None
        self.buffer = None
        
    def setup_environment(self):
        """Setup environment"""
        print("Setting up environment...")
        self.env = RemoteEnv.make(
            address=(self.args.env_address, self.args.env_port), 
            authkey=self.args.env_authkey.encode()
        )
        self.env = self.env.unwrapped
        self.env.reset()
        print(f"Environment setup complete, parallel environments: {self.env.num_envs}")
        
    def setup_agent(self):
        """Setup agent"""
        print("Setting up agent...")
        obs, infos = self.env.reset()
        obs = obs['policy']
        if "ee_pose" in infos:
            obs["ee_pose"] = infos["ee_pose"]
        
        self.agent = PPO(
            self.env, 
            process_maniskill_ppo_observation_override(copy.deepcopy(obs)), 
            self.args.ppo, 
            self.args.device, 
            train=False
        )
        
        assert self.args.checkpoint is not None, "Checkpoint is required"
        if self.args.checkpoint:
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
        obs, infos = self.env.reset()
        obs = obs['policy']
        if "ee_pose" in infos:
            obs["ee_pose"] = infos["ee_pose"]
        
        step_count = 0
        success_count = 0
        episode_count = 0
        
        # Create progress bar
        pbar = tqdm(total=self.args.num_steps, desc="Data collection progress")
        
        with torch.inference_mode():
            while step_count < self.args.num_steps:
                # Get actions
                actions = self.agent.agent.get_action(
                    process_maniskill_ppo_observation_override(copy.deepcopy(obs)), 
                    deterministic=self.args.deterministic
                )
                
                # Execute environment step
                next_obs, rewards, terminations, truncations, infos = self.env.step(actions)
                # add eef if in infos
                if "ee_pose" in infos:
                    next_obs["policy"]["ee_pose"] = infos["ee_pose"]
                next_obs = next_obs['policy']
                
                # Statistics
                success_count += infos['is_success'].sum().item()
                episode_count += (terminations | truncations).sum().item()
                
                # Prepare copies for storage with image conversion (keys unchanged)
                state_to_store = _convert_images_to_uint8_chw(copy.deepcopy(obs))
                next_state_to_store = _convert_images_to_uint8_chw(copy.deepcopy(next_obs))

                # Create transition data
                parallel_transition = BatchTransition(
                    state=state_to_store,
                    action=actions,
                    reward=rewards,
                    next_state=next_state_to_store,
                    done=terminations,
                    truncated=truncations,
                    complementary_info={"is_success": infos['is_success'].to(torch.float32)},
                )
                
                # Move to storage device
                tr = move_transition_to_device(parallel_transition, device=self.buffer.storage_device)
                
                # Add to buffer
                self.buffer.add(
                    state=tr["state"],
                    action=tr["action"],
                    reward=tr["reward"],
                    next_state=tr["next_state"],
                    done=tr["done"],
                    truncated=tr["truncated"],
                    complementary_info=tr["complementary_info"],
                )
                
                obs = next_obs
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
                       default="lerobot_liftobj_visual_hilserl_play", # read checkpoint from this config
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
    
    return parser.parse_args()


def main():
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
    collector = DataCollector(collection_args)
    dataset = collector.run_collection()
    
    print(f"\nDataset saved to: {collection_args.root_dir}")
    print(f"Dataset ID: {collection_args.repo_id}")
    print(f"Task name: {collection_args.task_name}")


if __name__ == "__main__":
    main()
