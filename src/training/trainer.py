"""
Trainer

Main training loop for any-order masked training.
Handles the single-pass masked reconstruction training.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from transformers import get_cosine_schedule_with_warmup
import wandb
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Optional, List
import numpy as np
import json

from ..masking.cell_masker import CellMasker, ScheduledCellMasker
from ..masking.attribute_masker import AttributeMasker
from .loss import MaskedReconstructionLoss
from ..evaluation.metrics import compute_metrics


class AnyOrderTrainer:
    """
    Trainer for any-order masked training paradigm.
    
    Handles:
    - Mask sampling across epochs (any-order learning)
    - Single-pass masked reconstruction
    - Training loop with evaluation
    - Checkpointing and logging
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        masker,  # CellMasker or AttributeMasker
        config: Dict,
        device: str = 'cuda',
        output_dir: str = 'outputs',
    ):
        """
        Args:
            model: LLaDA2.0 model (or other masked DLM)
            train_loader: Training data loader
            val_loader: Validation data loader
            masker: Mask sampler instance
            config: Training configuration
            device: Device to train on
            output_dir: Directory for checkpoints and logs
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.masker = masker
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        
        # Create output directories
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.log_dir = self.output_dir / 'logs'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup loss function
        self.loss_fn = MaskedReconstructionLoss(config)
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        
        # Setup scheduler
        total_steps = len(train_loader) * config['training']['num_epochs']
        self.scheduler = self._setup_scheduler(total_steps)
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # Setup logging
        if 'wandb' in config['training'].get('report_to', []):
            wandb.init(
                project=config['experiment']['project'],
                name=config['experiment']['name'],
                config=config,
                tags=config['experiment'].get('tags', []),
            )
    
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup AdamW optimizer."""
        train_config = self.config['training']
        
        # No weight decay for bias and layer norm
        no_decay = ['bias', 'LayerNorm.weight', 'layernorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': train_config['weight_decay'],
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            },
        ]
        
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=train_config['learning_rate'],
            betas=(train_config['adam_beta1'], train_config['adam_beta2']),
            eps=train_config['adam_epsilon'],
        )
        
        return optimizer
    
    def _setup_scheduler(self, total_steps: int):
        """Setup learning rate scheduler."""
        train_config = self.config['training']
        warmup_steps = train_config['warmup_steps']
        
        if train_config['lr_scheduler'] == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
        elif train_config['lr_scheduler'] == 'linear':
            scheduler = LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
        else:
            scheduler = None
        
        return scheduler
    
    def train(self):
        """Main training loop."""
        num_epochs = self.config['training']['num_epochs']
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train for one epoch
            train_metrics = self.train_epoch()
            
            # Evaluate
            if (epoch + 1) % self.config['training'].get('eval_every', 1) == 0:
                val_metrics = self.evaluate()
                
                # Log metrics
                self._log_metrics(train_metrics, val_metrics, epoch)
                
                # Save checkpoint if best
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.save_checkpoint('best')
                
                # Save periodic checkpoint
                if (epoch + 1) % self.config['training'].get('save_every', 5) == 0:
                    self.save_checkpoint(f'epoch_{epoch+1}')
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        total_obs_loss = 0
        total_act_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch+1}')
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            observations = batch['observations'].to(self.device)
            actions = batch['actions'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Update masker step (for scheduled masking)
            if hasattr(self.masker, 'update_step'):
                self.masker.update_step(self.global_step)
            
            # Sample masks (any-order masking!)
            mask_dict = self.masker.sample_mask(observations, actions, attention_mask)
            obs_mask = mask_dict['obs_mask']
            act_mask = mask_dict['act_mask']
            
            # Apply masks
            if hasattr(self.masker, 'apply_mask'):
                masked_obs, masked_act = self.masker.apply_mask(
                    observations, actions, obs_mask, act_mask
                )
            else:
                # Default masking: replace with special token
                masked_obs = observations.clone()
                masked_act = actions.clone()
                masked_obs[obs_mask.bool()] = self.config.get('mask_token_id', -1)
                masked_act[act_mask.bool()] = self.config.get('mask_token_id', -1)
            
            # Forward pass - single-pass masked reconstruction
            # Note: This is a simplified version. Actual implementation needs
            # to integrate with LLaDA2.0's API properly
            outputs = self.model(
                observations=masked_obs,
                actions=masked_act,
                attention_mask=attention_mask,
            )
            
            # Compute loss
            loss_dict = self.loss_fn(
                predictions=outputs,
                targets={
                    'observations': observations,
                    'actions': actions,
                },
                masks={
                    'obs_mask': obs_mask,
                    'act_mask': act_mask,
                },
            )
            
            loss = loss_dict['total_loss']
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config['training'].get('max_grad_norm'):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['max_grad_norm']
                )
            
            # Optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            total_obs_loss += loss_dict.get('obs_loss', 0)
            total_act_loss += loss_dict.get('act_loss', 0)
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'lr': self.optimizer.param_groups[0]['lr'],
                'mask_prob': self.masker.get_current_mask_prob(self.global_step)
                if hasattr(self.masker, 'get_current_mask_prob') else self.masker.mask_prob,
            })
            
            # Log to wandb
            if self.global_step % self.config['training']['logging_steps'] == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/obs_loss': loss_dict.get('obs_loss', 0),
                    'train/act_loss': loss_dict.get('act_loss', 0),
                    'train/lr': self.optimizer.param_groups[0]['lr'],
                    'train/mask_prob': self.masker.get_current_mask_prob(self.global_step)
                    if hasattr(self.masker, 'get_current_mask_prob') else self.masker.mask_prob,
                    **{f'train/{k}': v for k, v in mask_dict['mask_stats'].items()},
                }, step=self.global_step)
            
            self.global_step += 1
        
        return {
            'loss': total_loss / num_batches,
            'obs_loss': total_obs_loss / num_batches,
            'act_loss': total_act_loss / num_batches,
        }
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set."""
        self.model.eval()
        
        total_loss = 0
        total_obs_loss = 0
        total_act_loss = 0
        num_batches = 0
        
        all_predictions = []
        all_targets = []
        
        for batch in tqdm(self.val_loader, desc='Evaluating'):
            # Move batch to device
            observations = batch['observations'].to(self.device)
            actions = batch['actions'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Sample masks
            mask_dict = self.masker.sample_mask(observations, actions, attention_mask)
            obs_mask = mask_dict['obs_mask']
            act_mask = mask_dict['act_mask']
            
            # Apply masks
            if hasattr(self.masker, 'apply_mask'):
                masked_obs, masked_act = self.masker.apply_mask(
                    observations, actions, obs_mask, act_mask
                )
            else:
                masked_obs = observations.clone()
                masked_act = actions.clone()
                masked_obs[obs_mask.bool()] = self.config.get('mask_token_id', -1)
                masked_act[act_mask.bool()] = self.config.get('mask_token_id', -1)
            
            # Forward pass
            outputs = self.model(
                observations=masked_obs,
                actions=masked_act,
                attention_mask=attention_mask,
            )
            
            # Compute loss
            loss_dict = self.loss_fn(
                predictions=outputs,
                targets={
                    'observations': observations,
                    'actions': actions,
                },
                masks={
                    'obs_mask': obs_mask,
                    'act_mask': act_mask,
                },
            )
            
            total_loss += loss_dict['total_loss'].item()
            total_obs_loss += loss_dict.get('obs_loss', 0)
            total_act_loss += loss_dict.get('act_loss', 0)
            num_batches += 1
            
            # Store for metrics computation
            all_predictions.append(outputs)
            all_targets.append({
                'observations': observations,
                'actions': actions,
            })
        
        # Compute additional metrics
        metrics = compute_metrics(all_predictions, all_targets)
        
        metrics.update({
            'loss': total_loss / num_batches,
            'obs_loss': total_obs_loss / num_batches,
            'act_loss': total_act_loss / num_batches,
        })
        
        return metrics
    
    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / f'{name}.pt'
        
        torch.save({
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }, checkpoint_path)
        
        print(f'Saved checkpoint to {checkpoint_path}')
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f'Loaded checkpoint from {checkpoint_path}')
    
    def _log_metrics(self, train_metrics: Dict, val_metrics: Dict, epoch: int):
        """Log metrics to wandb and console."""
        print(f"\nEpoch {epoch+1}:")
        print(f"  Train Loss: {train_metrics['loss']:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}")
        
        if 'wandb' in self.config['training'].get('report_to', []):
            wandb.log({
                'epoch': epoch,
                **{f'train/{k}': v for k, v in train_metrics.items()},
                **{f'val/{k}': v for k, v in val_metrics.items()},
            }, step=self.global_step)
