"""
Mask Sampler

Core logic for sampling masks across trajectory elements.
Implements the any-order masking paradigm.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from abc import ABC, abstractmethod


class MaskSampler(ABC):
    """
    Abstract base class for mask sampling strategies.
    """
    
    def __init__(
        self,
        mask_prob: float = 0.30,
        mask_obs: bool = True,
        mask_act: bool = True,
        seed: Optional[int] = None,
    ):
        """
        Args:
            mask_prob: Base probability of masking
            mask_obs: Whether to mask observations
            mask_act: Whether to mask actions  
            seed: Random seed for reproducibility
        """
        self.mask_prob = mask_prob
        self.mask_obs = mask_obs
        self.mask_act = mask_act
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        self.rng = np.random.RandomState(seed)
    
    @abstractmethod
    def sample_mask(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Sample masks for a batch of trajectories.
        
        Args:
            observations: (batch, T, 7, 7, 3) observation tensor
            actions: (batch, T) action tensor
            attention_mask: (batch, T) attention mask
            
        Returns:
            Dictionary containing:
            - obs_mask: (batch, T, 7, 7) binary mask for observations
            - act_mask: (batch, T) binary mask for actions
            - mask_stats: Dictionary with masking statistics
        """
        raise NotImplementedError
    
    def get_current_mask_prob(self, step: int = 0) -> float:
        """
        Get current masking probability (for scheduled masking).
        
        Args:
            step: Current training step
            
        Returns:
            Current masking probability
        """
        return self.mask_prob
    
    def update_step(self, step: int):
        """
        Update internal state based on training step.
        Used for scheduled masking.
        """
        pass


class RandomMaskSampler(MaskSampler):
    """
    Random mask sampler - independently masks each element with probability p.
    This is the baseline any-order masking strategy.
    """
    
    def sample_mask(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Sample random masks for observations and actions.
        """
        batch_size, T, h, w, c = observations.shape
        device = observations.device
        
        # Sample observation masks (batch, T, h, w)
        if self.mask_obs:
            # For each position in the grid, decide whether to mask
            obs_mask_probs = torch.rand(batch_size, T, h, w, device=device)
            obs_mask = (obs_mask_probs < self.mask_prob).float()
            
            # Apply attention mask - don't mask padding
            obs_mask = obs_mask * attention_mask.unsqueeze(-1).unsqueeze(-1)
        else:
            obs_mask = torch.zeros(batch_size, T, h, w, device=device)
        
        # Sample action masks (batch, T)
        if self.mask_act:
            act_mask_probs = torch.rand(batch_size, T, device=device)
            act_mask = (act_mask_probs < self.mask_prob).float()
            
            # Apply attention mask
            act_mask = act_mask * attention_mask
        else:
            act_mask = torch.zeros(batch_size, T, device=device)
        
        # Compute statistics
        total_obs_positions = (attention_mask.sum() * h * w).item()
        total_act_positions = attention_mask.sum().item()
        
        masked_obs = obs_mask.sum().item()
        masked_act = act_mask.sum().item()
        
        mask_stats = {
            'obs_mask_ratio': masked_obs / (total_obs_positions + 1e-8),
            'act_mask_ratio': masked_act / (total_act_positions + 1e-8),
            'total_masked': masked_obs + masked_act,
            'total_positions': total_obs_positions + total_act_positions,
            'overall_mask_ratio': (masked_obs + masked_act) / (total_obs_positions + total_act_positions + 1e-8),
        }
        
        return {
            'obs_mask': obs_mask,
            'act_mask': act_mask,
            'mask_stats': mask_stats,
        }


class ScheduledMaskSampler(MaskSampler):
    """
    Scheduled mask sampler - gradually changes masking probability during training.
    Implements curriculum learning.
    """
    
    def __init__(
        self,
        mask_prob: float = 0.15,
        mask_obs: bool = True,
        mask_act: bool = True,
        schedule_type: str = 'linear',  # 'linear', 'cosine', 'step'
        start_prob: float = 0.15,
        end_prob: float = 0.50,
        schedule_steps: int = 10000,
        seed: Optional[int] = None,
    ):
        super().__init__(mask_prob, mask_obs, mask_act, seed)
        
        self.schedule_type = schedule_type
        self.start_prob = start_prob
        self.end_prob = end_prob
        self.schedule_steps = schedule_steps
        self.current_step = 0
        
        # Override initial mask_prob
        self.mask_prob = start_prob
    
    def get_current_mask_prob(self, step: Optional[int] = None) -> float:
        """
        Compute current masking probability based on schedule.
        """
        if step is None:
            step = self.current_step
        
        # Compute progress (0 to 1)
        progress = min(step / self.schedule_steps, 1.0)
        
        if self.schedule_type == 'linear':
            prob = self.start_prob + (self.end_prob - self.start_prob) * progress
        elif self.schedule_type == 'cosine':
            # Cosine schedule: smooth transition
            prob = self.start_prob + (self.end_prob - self.start_prob) * (1 - np.cos(progress * np.pi)) / 2
        elif self.schedule_type == 'step':
            # Step schedule: sudden jumps at 0.33, 0.66
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
        """
        Update current step and masking probability.
        """
        self.current_step = step
        self.mask_prob = self.get_current_mask_prob(step)
    
    def sample_mask(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Sample masks using current probability from schedule.
        """
        # Use the same logic as RandomMaskSampler but with scheduled probability
        batch_size, T, h, w, c = observations.shape
        device = observations.device
        
        current_prob = self.mask_prob  # Already updated by update_step
        
        # Sample observation masks
        if self.mask_obs:
            obs_mask_probs = torch.rand(batch_size, T, h, w, device=device)
            obs_mask = (obs_mask_probs < current_prob).float()
            obs_mask = obs_mask * attention_mask.unsqueeze(-1).unsqueeze(-1)
        else:
            obs_mask = torch.zeros(batch_size, T, h, w, device=device)
        
        # Sample action masks
        if self.mask_act:
            act_mask_probs = torch.rand(batch_size, T, device=device)
            act_mask = (act_mask_probs < current_prob).float()
            act_mask = act_mask * attention_mask
        else:
            act_mask = torch.zeros(batch_size, T, device=device)
        
        # Compute statistics
        total_obs_positions = (attention_mask.sum() * h * w).item()
        total_act_positions = attention_mask.sum().item()
        
        masked_obs = obs_mask.sum().item()
        masked_act = act_mask.sum().item()
        
        mask_stats = {
            'obs_mask_ratio': masked_obs / (total_obs_positions + 1e-8),
            'act_mask_ratio': masked_act / (total_act_positions + 1e-8),
            'total_masked': masked_obs + masked_act,
            'total_positions': total_obs_positions + total_act_positions,
            'overall_mask_ratio': (masked_obs + masked_act) / (total_obs_positions + total_act_positions + 1e-8),
            'current_mask_prob': current_prob,
            'schedule_progress': min(self.current_step / self.schedule_steps, 1.0),
        }
        
        return {
            'obs_mask': obs_mask,
            'act_mask': act_mask,
            'mask_stats': mask_stats,
        }


def create_mask_sampler(config: Dict) -> MaskSampler:
    """
    Factory function to create mask sampler from config.
    
    Args:
        config: Masking configuration dictionary
        
    Returns:
        MaskSampler instance
    """
    if config.get('mask_schedule') is not None:
        return ScheduledMaskSampler(
            mask_prob=config.get('mask_prob', 0.30),
            mask_obs=config.get('mask_obs', True),
            mask_act=config.get('mask_act', True),
            schedule_type=config['mask_schedule'],
            start_prob=config.get('schedule_start_prob', 0.15),
            end_prob=config.get('schedule_end_prob', 0.50),
            schedule_steps=config.get('schedule_steps', 10000),
            seed=config.get('seed', None),
        )
    else:
        return RandomMaskSampler(
            mask_prob=config.get('mask_prob', 0.30),
            mask_obs=config.get('mask_obs', True),
            mask_act=config.get('mask_act', True),
            seed=config.get('seed', None),
        )
