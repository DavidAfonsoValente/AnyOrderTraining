"""
Cell-Level Masker

Masks entire cells (all 3 attributes together) in the MiniGrid observation.
This teaches spatial reasoning, object permanence, and consistency across time.
"""

import torch
import numpy as np
from typing import Dict, Tuple
from .mask_sampler import MaskSampler


class CellMasker(MaskSampler):
    """
    Cell-level masking strategy.
    
    Masks all 3 attributes of a cell together:
    [cell_i_obj, cell_i_color, cell_i_state] → [MASK, MASK, MASK]
    
    This is coarser-grained than attribute-level masking but simpler.
    """
    
    def __init__(
        self,
        mask_prob: float = 0.30,
        mask_obs: bool = True,
        mask_act: bool = True,
        mask_all_attributes: bool = True,
        seed: int = None,
    ):
        """
        Args:
            mask_prob: Probability of masking a cell
            mask_obs: Whether to mask observations
            mask_act: Whether to mask actions
            mask_all_attributes: If True, mask all 3 attributes together (recommended)
            seed: Random seed
        """
        super().__init__(mask_prob, mask_obs, mask_act, seed)
        self.mask_all_attributes = mask_all_attributes
    
    def sample_mask(
        self,
        observations: torch.Tensor,  # (batch, T, 7, 7, 3)
        actions: torch.Tensor,  # (batch, T)
        attention_mask: torch.Tensor,  # (batch, T)
    ) -> Dict[str, torch.Tensor]:
        """
        Sample cell-level masks.
        
        Returns:
            Dictionary with:
            - obs_mask: (batch, T, 7, 7, 3) binary mask (1=mask, 0=keep)
            - act_mask: (batch, T) binary mask
            - mask_stats: Statistics dictionary
        """
        batch_size, T, h, w, c = observations.shape
        device = observations.device
        
        # Sample cell masks (batch, T, h, w)
        if self.mask_obs:
            cell_mask_probs = torch.rand(batch_size, T, h, w, device=device)
            cell_mask = (cell_mask_probs < self.mask_prob).float()
            
            # Apply attention mask - don't mask padding
            cell_mask = cell_mask * attention_mask.unsqueeze(-1).unsqueeze(-1)
            
            if self.mask_all_attributes:
                # Expand cell mask to all 3 attributes
                # (batch, T, h, w) -> (batch, T, h, w, 3)
                obs_mask = cell_mask.unsqueeze(-1).expand(-1, -1, -1, -1, c)
            else:
                # Only mask spatial positions, not attributes
                # (can be used for different variants)
                obs_mask = cell_mask.unsqueeze(-1)
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
        total_cells = (attention_mask.sum() * h * w).item()
        total_act_positions = attention_mask.sum().item()
        
        # Count masked cells (unique cells, not attributes)
        masked_cells = cell_mask.sum().item() if self.mask_obs else 0
        masked_act = act_mask.sum().item()
        
        mask_stats = {
            'cell_mask_ratio': masked_cells / (total_cells + 1e-8),
            'act_mask_ratio': masked_act / (total_act_positions + 1e-8),
            'masked_cells': masked_cells,
            'masked_actions': masked_act,
            'total_cells': total_cells,
            'total_actions': total_act_positions,
            'masking_strategy': 'cell-level',
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
        # Set masked positions to mask_token_id
        masked_obs[obs_mask.bool()] = mask_token_id
        
        # Apply action mask
        masked_act[act_mask.bool()] = mask_token_id
        
        return masked_obs, masked_act


class ScheduledCellMasker(CellMasker):
    """
    Cell-level masker with scheduled masking probability.
    Combines cell-level masking with curriculum learning.
    """
    
    def __init__(
        self,
        mask_prob: float = 0.15,
        mask_obs: bool = True,
        mask_act: bool = True,
        mask_all_attributes: bool = True,
        schedule_type: str = 'linear',
        start_prob: float = 0.15,
        end_prob: float = 0.50,
        schedule_steps: int = 10000,
        seed: int = None,
    ):
        super().__init__(mask_prob, mask_obs, mask_act, mask_all_attributes, seed)
        
        self.schedule_type = schedule_type
        self.start_prob = start_prob
        self.end_prob = end_prob
        self.schedule_steps = schedule_steps
        self.current_step = 0
        
        # Override initial mask_prob
        self.mask_prob = start_prob
    
    def get_current_mask_prob(self, step: int = None) -> float:
        """
        Compute current masking probability based on schedule.
        """
        if step is None:
            step = self.current_step
        
        progress = min(step / self.schedule_steps, 1.0)
        
        if self.schedule_type == 'linear':
            prob = self.start_prob + (self.end_prob - self.start_prob) * progress
        elif self.schedule_type == 'cosine':
            prob = self.start_prob + (self.end_prob - self.start_prob) * (1 - np.cos(progress * np.pi)) / 2
        elif self.schedule_type == 'step':
            if progress < 0.33:
                prob = self.start_prob
            elif progress < 0.66:
                prob = (self.start_prob + self.end_prob) / 2
            else:
                prob = self.end_prob
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")
        
        return prob
    
    def update_step(self, step: int):
        """Update current step and masking probability."""
        self.current_step = step
        self.mask_prob = self.get_current_mask_prob(step)
