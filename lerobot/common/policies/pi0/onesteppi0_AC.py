
from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.utils import cycle
from lerobot.common.envs.factory import make_env
from lerobot.common.optim.factory import make_optimizer_and_scheduler
from lerobot.common.policies.factory import make_policy
# from lerobot.common.policies.factory import register_policy_class
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy, PI0FlowMatching
from lerobot.common.policies.pi0.configuration_pi0 import PI0Config
from lerobot.common.constants import ACTION, OBS_ROBOT
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.common.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    has_method,
    init_logging,
)
from lerobot.common.utils.wandb_utils import WandBLogger
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.eval import eval_policy
from lerobot.configs.policies import PreTrainedConfig

import logging
import time
import datetime as dt
from pathlib import Path
import os
from contextlib import nullcontext
from dataclasses import dataclass, field
from pprint import pformat
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.amp import GradScaler
from torch.optim import Optimizer
from termcolor import colored


@PreTrainedConfig.register_subclass("pi0_onestep_ac")
@dataclass
class PI0OneStepACConfig(PI0Config):
    """Configuration for Pi0 one-step model."""
    type: str = "pi0_onestep_ac"



def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks
class PI0OneStepModel(nn.Module):
    """One-step distilled version of Pi0 that directly predicts actions"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # We use the same architecture as PI0FlowMatching
        self.flow_model = PI0FlowMatching(config)
        
        # Add a final projection layer to map from flow outputs to direct actions
        # This helps the model adapt from predicting noise residuals to predicting actions directly
        self.final_proj = nn.Linear(config.max_action_dim, config.max_action_dim)
        
        
    def forward(self, images, img_masks, lang_tokens, lang_masks, state):
        """Direct prediction of actions without iterative denoising"""
        bsize = state.shape[0]
        device = state.device
        
        # We still need some noise input for the architecture to work consistently
        # But we'll use a fixed time step (t=0) to represent "clean" actions
        fixed_time = torch.zeros(bsize, dtype=torch.float32, device=device)
        
        # Create a zero tensor with the shape of actions
        # This represents our "prediction" at t=0, which we'll update
        action_shape = (bsize, self.config.n_action_steps, self.config.max_action_dim)
        x_0 = torch.zeros(action_shape, dtype=torch.float32, device=device)
        
        # Process prefix (images and language)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.flow_model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Get cached KV from prefix processing
        _, past_key_values = self.flow_model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            fill_kv_cache=True,
        )
        
        # Process state and current action prediction
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.flow_model.embed_suffix(
            state, x_0, fixed_time
        )

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Get model outputs
        outputs_embeds, _ = self.flow_model.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=True,
            fill_kv_cache=False,
        )
        
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.n_action_steps:]
        suffix_out = suffix_out.to(dtype=torch.float32)
        
        # Instead of using this as a flow/velocity prediction,
        # we transform it into a direct action prediction
        actions = self.flow_model.action_out_proj(suffix_out)
        actions = self.final_proj(actions)  # Additional adaptation layer
        
        return actions


class PI0OneStepModelCritic(nn.Module):
    """One-step distilled version of Pi0 that directly predicts actions"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # We use the same architecture as PI0FlowMatching
        self.flow_model = PI0FlowMatching(config)
        
        # Add a final projection layer to map from flow outputs to direct actions
        # This helps the model adapt from predicting noise residuals to predicting actions directly
        self.lienar_prob = nn.Linear(config.max_action_dim, 1)
        
        
    def forward(self, images, img_masks, lang_tokens, lang_masks, state,action):
        """Direct prediction of actions without iterative denoising"""
        bsize = state.shape[0]
        device = state.device
        
        # We still need some noise input for the architecture to work consistently
        # But we'll use a fixed time step (t=0) to represent "clean" actions
        fixed_time = torch.ones(bsize, dtype=torch.float32, device=device)
        
        # Create a zero tensor with the shape of actions
        # This represents our "prediction" at t=0, which we'll update
        action_shape = (bsize, self.config.n_action_steps, self.config.max_action_dim)
        x_0 = action
        
        # Process prefix (images and language)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.flow_model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Get cached KV from prefix processing
        _, past_key_values = self.flow_model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            fill_kv_cache=True,
        )
        
        # Process state and current action prediction
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.flow_model.embed_suffix(
            state, x_0, fixed_time
        )

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Get model outputs
        outputs_embeds, _ = self.flow_model.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=True,
            fill_kv_cache=False,
        )
        
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.n_action_steps:]
        suffix_out = suffix_out.to(dtype=torch.float32)
        
        # Instead of using this as a flow/velocity prediction,
        # we transform it into a direct action prediction
        actions = self.flow_model.action_out_proj(suffix_out)
        score = self.lienar_prob(actions)  # Additional adaptation layer
        
        return score

class PI0OneStepACPolicy(PI0Policy):
    """Wrapper around PI0OneStepModel to use the same interface as PI0Policy"""
    
    config_class = PI0OneStepACConfig
    name = "pi0_onestep_ac"
    
    def __init__(self, config, dataset_stats=None, teacher_policy=None):
        # Initialize the parent class but we'll override the model
        super(PI0Policy, self).__init__(config)  # Call grandparent's init
        config.validate_features()
        self.config = config
        
        for input_feature in config.input_features:
            config["next_" + input_feature] = config[input_feature]
        
        
        # Setup normalization components
        if hasattr(config, 'normalization_mapping') and hasattr(config, 'input_features') and hasattr(config, 'output_features'):
            from lerobot.common.policies.normalize import Normalize, Unnormalize
            self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
            self.normalize_targets = Normalize(config.output_features, config.normalization_mapping, dataset_stats)
            self.unnormalize_outputs = Unnormalize(config.output_features, config.normalization_mapping, dataset_stats)
            
            
        
        # Setup tokenizer
        from transformers import AutoTokenizer
        self.language_tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
        
        # Create our one-step model instead of flow matching
        self.model = PI0OneStepModel(config)
        
        # Store the teacher for distillation
        self.teacher_policy = teacher_policy
        
        # Reset action queue
        self.reset()
        
    def make_critic(self):
        """Create the critic model"""
        # Use the same architecture as PI0FlowMatching
        self.critic = PI0OneStepModelCritic(self.config)
        missing_keys, unexpected_keys = self.critic.load_state_dict(self.model.state_dict(), strict=False)
        print("Missing keys:", missing_keys)
        print("Unexpected keys:", unexpected_keys)
        
        # Initialize the target critic
        self.target_critic = PI0OneStepModelCritic(self.config)
        
        # Copy weights from critic to target critic
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # Set target critic to eval mode
        self.target_critic.eval()
    
    @torch.no_grad()
    def select_action(self, batch, noise=None):
        """Select actions directly without iterative denoising"""
        self.eval()
        
        if self.config.adapt_to_pi_aloha:
            batch[OBS_ROBOT] = self._pi_aloha_decode_state(batch[OBS_ROBOT])
            
        batch = self.normalize_inputs(batch)
        
        # Action queue logic for n_action_steps > 1
        if len(self._action_queue) == 0:
            images, img_masks = self.prepare_images(batch)
            state = self.prepare_state(batch)
            lang_tokens, lang_masks = self.prepare_language(batch)
            
            # Directly predict actions
            actions = self.model(images, img_masks, lang_tokens, lang_masks, state)
            
            # Unpad actions
            original_action_dim = self.config.action_feature.shape[0]
            actions = actions[:, :, :original_action_dim]
            
            actions = self.unnormalize_outputs({"action": actions})["action"]
            
            if self.config.adapt_to_pi_aloha:
                actions = self._pi_aloha_encode_actions(actions)
                
            # Populate queue
            self._action_queue.extend(actions.transpose(0, 1))
            
        return self._action_queue.popleft()
    
    def prepare_next_images(self, batch):
        """Apply Pi0 preprocessing to the images, like resizing to 224x224 and padding to keep aspect ratio, and
        convert pixel range from [0.0, 1.0] to [-1.0, 1.0] as requested by SigLIP.
        """
        images = []
        img_masks = []

        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. (batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )

        # Preprocess image features present in the batch
        for key in present_img_keys:
            img = batch[key]

            if self.config.resize_imgs_with_padding is not None:
                img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)

            # Normalize from range [0,1] to [-1,1] as expacted by siglip
            img = img * 2.0 - 1.0

            bsize = img.shape[0]
            device = img.device
            mask = torch.ones(bsize, dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)

        # Create image features not present in the batch
        # as fully 0 padded images.
        for num_empty_cameras in range(len(missing_img_keys)):
            if num_empty_cameras >= self.config.empty_cameras:
                break
            img = torch.ones_like(img) * -1
            mask = torch.zeros_like(mask)
            images.append(img)
            img_masks.append(mask)

        return images, img_masks
    def forward(self, batch, temperature=1.0, soft_weight=0.5, hard_weight=0.5):
        """Forward pass for training with distillation"""
        # Handle Pi-Aloha adaptation if needed
        if self.config.adapt_to_pi_aloha:
            batch[OBS_ROBOT] = self._pi_aloha_decode_state(batch[OBS_ROBOT])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])
            
        print(batch.keys())
        print( self.config.image_features)
        # print(batch["next_observation.state"].keys())
        # for key in batch:
        #     if key.startswith("observation.images."):
        #         batch["next_" + key] = batch[key]
        
        
        # Normalize inputs and targets
        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)
        
        
        
        # Prepare inputs
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens, lang_masks = self.prepare_language(batch)
        target_actions = self.prepare_action(batch)
        actions_is_pad = batch.get("actions_is_pad")
        
        # Get direct action predictions from student
        pred_actions = self.model(images, img_masks, lang_tokens, lang_masks, state)
        
        
        
        
        loss_dict = {}
        
        
        rewards = batch.get("reward")
        with torch.no_grad():
            pred_next_actions = self.model(next_images, next_img_masks, next_lang_tokens, next_lang_masks, next_state)
            pred_next_actions = torch.clamp(pred_next_actions, min=0.0, max=1.0)
            next_qs = self.target_critic(next_images, next_img_masks, next_lang_tokens, next_lang_masks, next_state, pred_next_actions)
            next_q = next_qs.mean(dim=0)
            target_q = rewards + self.discount * next_q
        q = self.critic(images, img_masks, lang_tokens, lang_masks, state, target_actions)
        
        critic_loss = F.mse_loss(q, target_q)
        
        loss_dict["critic_loss"] = critic_loss.mean().item()
        
        # Compute standard supervised loss with ground truth
        hard_loss = F.mse_loss(pred_actions, target_actions, reduction="none")
        # print("gt:",pred_actions.shape, target_actions.shape)
        
        # Get teacher predictions if teacher is available
        if self.teacher_policy is not None:
            with torch.no_grad():
                # Sample actions using the teacher model's flow matching 
                teacher_actions = self.teacher_policy.model.sample_actions(
                    images, img_masks, lang_tokens, lang_masks, state
                )
            
            # Compute distillation loss (student mimicking teacher)
            soft_loss = F.mse_loss(pred_actions, teacher_actions, reduction="none")
            teacher_loss = F.mse_loss(teacher_actions, target_actions, reduction="none")
            # print("teacher:",pred_actions.shape, teacher_actions.shape)
            
            # Combine the two losses
            losses = soft_weight * soft_loss + hard_weight * hard_loss
            loss_dict["soft_loss"] = soft_loss.mean().item()
            loss_dict["hard_loss"] = hard_loss.mean().item()
            loss_dict["teacher_loss"] = teacher_loss.mean().item()
        else:
            # If no teacher, just use the hard loss
            losses = hard_loss
            
        pred_actions = torch.clamp(pred_actions, min=0.0, max=1.0)
        qs = self.critic(images, img_masks, lang_tokens, lang_masks, state, pred_actions)
        q = -qs.mean(dim=0)
        losses = losses + -q
        loss_dict["q_loss"] = q.mean().item()
        
        # Handle padding
        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            losses = losses * in_episode_bound.unsqueeze(-1)
            
        # Remove padding
        original_action_dim = self.config.action_feature.shape[0]
        losses = losses[:, :, :original_action_dim]
        
        # Calculate final loss
        loss = losses.mean()
        loss_dict["total_loss"] = loss.item()
        
        return loss, loss_dict
