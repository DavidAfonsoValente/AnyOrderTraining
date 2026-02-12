"""
Loss Functions

Masked reconstruction losses for observations and actions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class MaskedReconstructionLoss(nn.Module):
    """
    Loss function for masked reconstruction.
    
    Computes cross-entropy loss only on masked positions.
    Supports separate losses for observations and actions.
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Loss weights
        self.obs_loss_weight = config.get('obs_loss_weight', 1.0)
        self.act_loss_weight = config.get('act_loss_weight', 1.0)
        
        # Whether to compute separate losses for attributes
        self.attribute_level_loss = config.get('attribute_level_loss', False)
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        masks: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute masked reconstruction loss.
        
        Args:
            predictions: Dictionary with:
                - 'observations': (batch, T, 7, 7, 3, num_classes) logits
                - 'actions': (batch, T, num_actions) logits
            targets: Dictionary with:
                - 'observations': (batch, T, 7, 7, 3) ground truth indices
                - 'actions': (batch, T) ground truth indices
            masks: Dictionary with:
                - 'obs_mask': (batch, T, 7, 7, 3) binary mask
                - 'act_mask': (batch, T) binary mask
                
        Returns:
            Dictionary with losses
        """
        obs_loss = 0.0
        act_loss = 0.0
        
        # Observation loss
        if 'observations' in predictions and 'observations' in targets:
            obs_logits = predictions['observations']  # (batch, T, 7, 7, 3, num_classes)
            obs_targets = targets['observations']  # (batch, T, 7, 7, 3)
            obs_mask = masks['obs_mask']  # (batch, T, 7, 7, 3)
            
            if self.attribute_level_loss:
                # Compute separate losses for each attribute
                obj_loss = self._compute_masked_ce_loss(
                    obs_logits[..., 0, :],  # Object logits
                    obs_targets[..., 0],  # Object targets
                    obs_mask[..., 0],  # Object mask
                )
                
                color_loss = self._compute_masked_ce_loss(
                    obs_logits[..., 1, :],  # Color logits
                    obs_targets[..., 1],  # Color targets
                    obs_mask[..., 1],  # Color mask
                )
                
                state_loss = self._compute_masked_ce_loss(
                    obs_logits[..., 2, :],  # State logits
                    obs_targets[..., 2],  # State targets
                    obs_mask[..., 2],  # State mask
                )
                
                obs_loss = (obj_loss + color_loss + state_loss) / 3.0
                
                # Return detailed losses
                attribute_losses = {
                    'obj_loss': obj_loss.item(),
                    'color_loss': color_loss.item(),
                    'state_loss': state_loss.item(),
                }
            else:
                # Compute overall observation loss
                # Reshape for cross-entropy
                batch_size, T, h, w, c, num_classes = obs_logits.shape
                
                obs_logits_flat = obs_logits.reshape(-1, num_classes)
                obs_targets_flat = obs_targets.reshape(-1)
                obs_mask_flat = obs_mask.reshape(-1)
                
                obs_loss = self._compute_masked_ce_loss(
                    obs_logits_flat,
                    obs_targets_flat,
                    obs_mask_flat,
                )
                
                attribute_losses = {}
        else:
            attribute_losses = {}
        
        # Action loss
        if 'actions' in predictions and 'actions' in targets:
            act_logits = predictions['actions']  # (batch, T, num_actions)
            act_targets = targets['actions']  # (batch, T)
            act_mask = masks['act_mask']  # (batch, T)
            
            act_loss = self._compute_masked_ce_loss(
                act_logits.reshape(-1, act_logits.shape[-1]),
                act_targets.reshape(-1),
                act_mask.reshape(-1),
            )
        
        # Combine losses
        total_loss = (
            self.obs_loss_weight * obs_loss +
            self.act_loss_weight * act_loss
        )
        
        return {
            'total_loss': total_loss,
            'obs_loss': obs_loss.item() if isinstance(obs_loss, torch.Tensor) else obs_loss,
            'act_loss': act_loss.item() if isinstance(act_loss, torch.Tensor) else act_loss,
            **attribute_losses,
        }
    
    def _compute_masked_ce_loss(
        self,
        logits: torch.Tensor,  # (N, num_classes)
        targets: torch.Tensor,  # (N,)
        mask: torch.Tensor,  # (N,)
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss only on masked positions.
        
        Args:
            logits: Prediction logits
            targets: Ground truth class indices
            mask: Binary mask (1=compute loss, 0=ignore)
            
        Returns:
            Scalar loss
        """
        # Ensure mask is boolean
        mask_bool = mask.bool()
        
        # Check if there are any masked positions
        if mask_bool.sum() == 0:
            return torch.tensor(0.0, device=logits.device)
        
        # Select only masked positions
        masked_logits = logits[mask_bool]
        masked_targets = targets[mask_bool]
        
        # Compute cross-entropy
        loss = F.cross_entropy(masked_logits, masked_targets, reduction='mean')
        
        return loss


class WorldModelLoss(MaskedReconstructionLoss):
    """
    Extended loss for world model learning.
    
    Adds additional losses for:
    - Next observation prediction
    - Reward prediction
    - Done prediction
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        self.predict_reward = config.get('predict_reward', False)
        self.predict_done = config.get('predict_done', False)
        self.reward_loss_weight = config.get('reward_loss_weight', 0.1)
        self.done_loss_weight = config.get('done_loss_weight', 0.1)
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        masks: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute world model losses.
        """
        # Get base reconstruction losses
        loss_dict = super().forward(predictions, targets, masks)
        
        # Add reward prediction loss if enabled
        if self.predict_reward and 'rewards' in predictions and 'rewards' in targets:
            reward_pred = predictions['rewards']  # (batch, T)
            reward_target = targets['rewards']  # (batch, T)
            attention_mask = targets.get('attention_mask', torch.ones_like(reward_target))
            
            reward_loss = F.mse_loss(
                reward_pred[attention_mask.bool()],
                reward_target[attention_mask.bool()],
                reduction='mean'
            )
            
            loss_dict['reward_loss'] = reward_loss.item()
            loss_dict['total_loss'] = loss_dict['total_loss'] + self.reward_loss_weight * reward_loss
        
        # Add done prediction loss if enabled
        if self.predict_done and 'dones' in predictions and 'dones' in targets:
            done_pred = predictions['dones']  # (batch, T)
            done_target = targets['dones']  # (batch, T)
            attention_mask = targets.get('attention_mask', torch.ones_like(done_target))
            
            done_loss = F.binary_cross_entropy_with_logits(
                done_pred[attention_mask.bool()],
                done_target[attention_mask.bool()],
                reduction='mean'
            )
            
            loss_dict['done_loss'] = done_loss.item()
            loss_dict['total_loss'] = loss_dict['total_loss'] + self.done_loss_weight * done_loss
        
        return loss_dict
