"""
Attribute-Level Masker

Masks individual attributes within cells.
This is more fine-grained and teaches dynamics, affordances, and causal structure.

Example: [door, red, MASK] → predict closed/open?
"""

import torch
import numpy as np
from typing import Dict, Tuple
from .mask_sampler import MaskSampler


class AttributeMasker(MaskSampler):
    """
    Attribute-level masking strategy.
    
    Masks individual attributes within cells:
    - Object type (e.g., door, key, ball)
    - Color (e.g., red, blue, green)
    - State (e.g., open, closed, locked)
    
    Each attribute can be masked independently with different probabilities.
    """
    
    def __init__(
        self,
        mask_prob: float = 0.30,
        mask_obs: bool = True,
        mask_act: bool = True,
        mask_object_prob: float = 0.33,
        mask_color_prob: float = 0.33,
        mask_state_prob: float = 0.33,
        seed: int = None,
    ):
        """
        Args:
            mask_prob: Overall base probability (can override specific probs)
            mask_obs: Whether to mask observations
            mask_act: Whether to mask actions
            mask_object_prob: Probability of masking object attribute
            mask_color_prob: Probability of masking color attribute
            mask_state_prob: Probability of masking state attribute
            seed: Random seed
        """
        super().__init__(mask_prob, mask_obs, mask_act, seed)
        
        self.mask_object_prob = mask_object_prob
        self.mask_color_prob = mask_color_prob
        self.mask_state_prob = mask_state_prob
    
    def sample_mask(
        self,
        observations: torch.Tensor,  # (batch, T, 7, 7, 3)
        actions: torch.Tensor,  # (batch, T)
        attention_mask: torch.Tensor,  # (batch, T)
    ) -> Dict[str, torch.Tensor]:
        """
        Sample attribute-level masks.
        
        Each of the 3 attributes (object, color, state) is masked independently.
        
        Returns:
            Dictionary with:
            - obs_mask: (batch, T, 7, 7, 3) binary mask
            - act_mask: (batch, T) binary mask
            - mask_stats: Statistics dictionary
        """
        batch_size, T, h, w, c = observations.shape
        device = observations.device
        
        assert c == 3, "Observations must have 3 attributes (object, color, state)"
        
        # Sample masks for each attribute independently
        if self.mask_obs:
            # Create separate masks for each attribute
            obj_mask_probs = torch.rand(batch_size, T, h, w, device=device)
            color_mask_probs = torch.rand(batch_size, T, h, w, device=device)
            state_mask_probs = torch.rand(batch_size, T, h, w, device=device)
            
            obj_mask = (obj_mask_probs < self.mask_object_prob).float()
            color_mask = (color_mask_probs < self.mask_color_prob).float()
            state_mask = (state_mask_probs < self.mask_state_prob).float()
            
            # Apply attention mask
            attention_expanded = attention_mask.unsqueeze(-1).unsqueeze(-1)  # (batch, T, 1, 1)
            obj_mask = obj_mask * attention_expanded
            color_mask = color_mask * attention_expanded
            state_mask = state_mask * attention_expanded
            
            # Stack into (batch, T, h, w, 3)
            obs_mask = torch.stack([obj_mask, color_mask, state_mask], dim=-1)
        else:
            obs_mask = torch.zeros(batch_size, T, h, w, c, device=device)
        
        # Sample action masks (same as base class)
        if self.mask_act:
            act_mask_probs = torch.rand(batch_size, T, device=device)
            act_mask = (act_mask_probs < self.mask_prob).float()
            act_mask = act_mask * attention_mask
        else:
            act_mask = torch.zeros(batch_size, T, device=device)
        
        # Compute statistics
        total_attributes = (attention_mask.sum() * h * w * c).item()
        total_act_positions = attention_mask.sum().item()
        
        masked_attributes = obs_mask.sum().item()
        masked_act = act_mask.sum().item()
        
        # Compute per-attribute statistics
        if self.mask_obs:
            masked_objects = obs_mask[..., 0].sum().item()
            masked_colors = obs_mask[..., 1].sum().item()
            masked_states = obs_mask[..., 2].sum().item()
            
            total_per_attribute = (attention_mask.sum() * h * w).item()
        else:
            masked_objects = masked_colors = masked_states = 0
            total_per_attribute = 1
        
        mask_stats = {
            'attribute_mask_ratio': masked_attributes / (total_attributes + 1e-8),
            'act_mask_ratio': masked_act / (total_act_positions + 1e-8),
            'masked_attributes': masked_attributes,
            'masked_actions': masked_act,
            'total_attributes': total_attributes,
            'total_actions': total_act_positions,
            'masking_strategy': 'attribute-level',
            # Per-attribute breakdown
            'masked_objects': masked_objects,
            'masked_colors': masked_colors,
            'masked_states': masked_states,
            'object_mask_ratio': masked_objects / (total_per_attribute + 1e-8),
            'color_mask_ratio': masked_colors / (total_per_attribute + 1e-8),
            'state_mask_ratio': masked_states / (total_per_attribute + 1e-8),
        }
        
        return {
            'obs_mask': obs_mask,
            'act_mask': act_mask,
            'mask_stats': mask_stats,
        }
    
    def apply_mask(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        obs_mask: torch.Tensor,
        act_mask: torch.Tensor,
        mask_token_id: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply masks to observations and actions.
        
        Args:
            observations: (batch, T, 7, 7, 3)
            actions: (batch, T)
            obs_mask: (batch, T, 7, 7, 3) binary mask
            act_mask: (batch, T) binary mask
            mask_token_id: Token ID to use for masked positions
            
        Returns:
            Tuple of masked observations and actions
        """
        # Clone to avoid in-place modification
        masked_obs = observations.clone()
        masked_act = actions.clone()
        
        # Apply observation mask
        masked_obs[obs_mask.bool()] = mask_token_id
        
        # Apply action mask
        masked_act[act_mask.bool()] = mask_token_id
        
        return masked_obs, masked_act


class AdaptiveAttributeMasker(AttributeMasker):
    """
    Adaptive attribute masker that adjusts masking probabilities based on
    the importance or difficulty of each attribute.
    
    For example, masking state (open/closed) might be more informative
    for learning dynamics than masking color.
    """
    
    def __init__(
        self,
        mask_prob: float = 0.30,
        mask_obs: bool = True,
        mask_act: bool = True,
        initial_object_prob: float = 0.25,
        initial_color_prob: float = 0.25,
        initial_state_prob: float = 0.50,  # Higher initial prob for state
        adapt_rate: float = 0.01,
        seed: int = None,
    ):
        """
        Args:
            adapt_rate: Rate at which to adapt probabilities (not yet implemented)
        """
        super().__init__(
            mask_prob=mask_prob,
            mask_obs=mask_obs,
            mask_act=mask_act,
            mask_object_prob=initial_object_prob,
            mask_color_prob=initial_color_prob,
            mask_state_prob=initial_state_prob,
            seed=seed,
        )
        
        self.adapt_rate = adapt_rate
        self.initial_probs = {
            'object': initial_object_prob,
            'color': initial_color_prob,
            'state': initial_state_prob,
        }
    
    def adapt_probabilities(self, losses: Dict[str, float]):
        """
        Adapt masking probabilities based on reconstruction losses.
        Higher loss → increase masking probability for that attribute.
        
        Args:
            losses: Dictionary with 'object_loss', 'color_loss', 'state_loss'
        """
        # Normalize losses
        total_loss = sum(losses.values()) + 1e-8
        
        # Adjust probabilities proportional to loss
        self.mask_object_prob = min(
            0.8,
            self.mask_object_prob + self.adapt_rate * (losses.get('object_loss', 0) / total_loss - 0.33)
        )
        self.mask_color_prob = min(
            0.8,
            self.mask_color_prob + self.adapt_rate * (losses.get('color_loss', 0) / total_loss - 0.33)
        )
        self.mask_state_prob = min(
            0.8,
            self.mask_state_prob + self.adapt_rate * (losses.get('state_loss', 0) / total_loss - 0.33)
        )
        
        # Ensure probabilities stay in valid range
        self.mask_object_prob = max(0.05, self.mask_object_prob)
        self.mask_color_prob = max(0.05, self.mask_color_prob)
        self.mask_state_prob = max(0.05, self.mask_state_prob)
