#!/usr/bin/env python3
"""
Main Training Script for Any-Order Masked Training

This script sets up the model, data, masker, and trainer, then runs training.
"""

import argparse
import yaml
import torch
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from torch.utils.data import DataLoader
from data.minigrid_dataset import BabyAIDataset, collate_fn
from masking.cell_masker import CellMasker, ScheduledCellMasker
from masking.attribute_masker import AttributeMasker
from training.trainer import AnyOrderTrainer


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Handle config inheritance (_base_)
    if '_base_' in config:
        base_path = Path(config_path).parent / config['_base_']
        base_config = load_config(str(base_path))
        
        # Merge configs (child overrides parent)
        def merge_dicts(base, override):
            result = base.copy()
            for key, value in override.items():
                if key == '_base_':
                    continue
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = merge_dicts(result[key], value)
                else:
                    result[key] = value
            return result
        
        config = merge_dicts(base_config, config)
    
    return config


def create_masker(config: dict):
    """Create mask sampler based on config."""
    masking_config = config['masking']
    strategy = masking_config['strategy']
    
    if strategy == 'cell':
        if masking_config.get('mask_schedule') is not None:
            return ScheduledCellMasker(
                mask_prob=masking_config['mask_prob'],
                mask_obs=masking_config['mask_obs'],
                mask_act=masking_config['mask_act'],
                mask_all_attributes=masking_config['cell_masking']['mask_all_attributes'],
                schedule_type=masking_config['mask_schedule'],
                start_prob=masking_config['schedule_start_prob'],
                end_prob=masking_config['schedule_end_prob'],
                schedule_steps=masking_config['schedule_steps'],
                seed=config['system']['seed'],
            )
        else:
            return CellMasker(
                mask_prob=masking_config['mask_prob'],
                mask_obs=masking_config['mask_obs'],
                mask_act=masking_config['mask_act'],
                mask_all_attributes=masking_config['cell_masking']['mask_all_attributes'],
                seed=config['system']['seed'],
            )
    
    elif strategy == 'attribute':
        return AttributeMasker(
            mask_prob=masking_config['mask_prob'],
            mask_obs=masking_config['mask_obs'],
            mask_act=masking_config['mask_act'],
            mask_object_prob=masking_config['attribute_masking']['mask_object_prob'],
            mask_color_prob=masking_config['attribute_masking']['mask_color_prob'],
            mask_state_prob=masking_config['attribute_masking']['mask_state_prob'],
            seed=config['system']['seed'],
        )
    
    else:
        raise ValueError(f"Unknown masking strategy: {strategy}")


def create_model(config: dict):
    """
    Create and initialize model.
    
    NOTE: This is a placeholder. You need to integrate with actual LLaDA2.0 model.
    For now, we'll create a simple mock model for testing the pipeline.
    """
    from torch import nn
    
    class MockModel(nn.Module):
        """Mock model for testing the training pipeline."""
        def __init__(self):
            super().__init__()
            # Observation decoder: predicts each attribute
            self.obs_decoder = nn.Sequential(
                nn.Linear(49 * 3, 256),
                nn.ReLU(),
                nn.Linear(256, 49 * 3 * 11),  # 49 cells, 3 attrs, 11 classes each
            )
            # Action decoder
            self.act_decoder = nn.Linear(49 * 3, 7)  # 7 possible actions
        
        def forward(self, observations, actions, attention_mask):
            batch_size, T = actions.shape
            
            # Flatten observations
            obs_flat = observations.reshape(batch_size, T, -1)  # (batch, T, 49*3)
            
            # Decode observations
            obs_logits = self.obs_decoder(obs_flat)  # (batch, T, 49*3*11)
            obs_logits = obs_logits.reshape(batch_size, T, 7, 7, 3, 11)
            
            # Decode actions
            act_logits = self.act_decoder(obs_flat)  # (batch, T, 7)
            
            return {
                'observations': obs_logits,
                'actions': act_logits,
            }
    
    print("WARNING: Using mock model. Replace with actual LLaDA2.0 integration!")
    return MockModel()


def main():
    parser = argparse.ArgumentParser(description='Train any-order masked agent')
    
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (overrides config)')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Data directory (overrides config)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    # Allow overriding config values from command line
    parser.add_argument('--masking.mask_prob', type=float, dest='mask_prob',
                        help='Override masking probability')
    parser.add_argument('--training.batch_size', type=int, dest='batch_size',
                        help='Override batch size')
    parser.add_argument('--training.learning_rate', type=float, dest='learning_rate',
                        help='Override learning rate')
    
    args = parser.parse_args()
    
    # Load config
    print(f"Loading config from {args.config}")
    config = load_config(args.config)
    
    # Override config with command line args
    if args.output_dir:
        config['training']['output_dir'] = args.output_dir
    if args.data_dir:
        config['data']['data_dir'] = args.data_dir
    if args.mask_prob is not None:
        config['masking']['mask_prob'] = args.mask_prob
    if args.batch_size is not None:
        config['data']['batch_size'] = args.batch_size
    if args.learning_rate is not None:
        config['training']['learning_rate'] = args.learning_rate
    
    # Set output directory
    output_dir = Path(config['training'].get('output_dir', 'outputs')) / config['experiment']['name']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_save_path = output_dir / 'config.yaml'
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Saved config to {config_save_path}")
    
    # Set seed
    torch.manual_seed(config['system']['seed'])
    
    # Create datasets
    print("\nCreating datasets...")
    data_dir = Path(config['data']['data_dir']) / config['data']['env_name'].replace('/', '_')
    
    train_dataset = BabyAIDataset(
        data_dir=str(data_dir),
        split='train',
        max_trajectory_length=config['data']['max_trajectory_length'],
    )
    
    val_dataset = BabyAIDataset(
        data_dir=str(data_dir),
        split='val',
        max_trajectory_length=config['data']['max_trajectory_length'],
    )
    
    print(f"Train dataset: {len(train_dataset)} trajectories")
    print(f"Val dataset: {len(val_dataset)} trajectories")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=config['data']['shuffle'],
        num_workers=config['data']['num_workers'],
        collate_fn=collate_fn,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        collate_fn=collate_fn,
    )
    
    # Create masker
    print("\nCreating masker...")
    masker = create_masker(config)
    print(f"Masking strategy: {config['masking']['strategy']}")
    print(f"Masking probability: {config['masking']['mask_prob']}")
    
    # Create model
    print("\nCreating model...")
    model = create_model(config)
    print(f"Model created")
    
    # Create trainer
    print("\nCreating trainer...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    trainer = AnyOrderTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        masker=masker,
        config=config,
        device=device,
        output_dir=str(output_dir),
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Start training
    print("\n" + "="*80)
    print("Starting Training")
    print("="*80 + "\n")
    
    trainer.train()
    
    print("\n" + "="*80)
    print("Training Complete!")
    print(f"Checkpoints saved to: {trainer.checkpoint_dir}")
    print(f"Logs saved to: {trainer.log_dir}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
