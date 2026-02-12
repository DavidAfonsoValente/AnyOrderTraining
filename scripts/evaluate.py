#!/usr/bin/env python3
"""
Evaluation Script

Evaluate trained models on various metrics.
"""

import argparse
import torch
import yaml
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from torch.utils.data import DataLoader
from data.minigrid_dataset import BabyAIDataset, collate_fn
from evaluation.metrics import (
    compute_metrics,
    compute_world_model_nll,
    evaluate_partial_observability_robustness,
    evaluate_task_success,
)


def load_model_and_config(checkpoint_path: str):
    """Load model and config from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']
    
    # Create model (using mock model for now)
    # TODO: Replace with actual LLaDA2.0 model loading
    from scripts.train import create_model
    model = create_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, config


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--metric', type=str, default='all',
                        choices=['all', 'world_model_nll', 'task_success', 
                                'robustness', 'accuracy'],
                        help='Metric to evaluate')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Data directory (for world_model_nll and accuracy)')
    parser.add_argument('--env', type=str, default=None,
                        help='Environment name (for task_success and robustness)')
    parser.add_argument('--num_episodes', type=int, default=100,
                        help='Number of episodes for environment-based eval')
    parser.add_argument('--occlusion_prob', type=float, default=0.3,
                        help='Occlusion probability for robustness eval')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    
    args = parser.parse_args()
    
    # Load model and config
    print(f"Loading checkpoint from {args.checkpoint}")
    model, config = load_model_and_config(args.checkpoint)
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    
    print(f"Using device: {device}")
    print(f"Evaluating metric: {args.metric}")
    
    results = {}
    
    # World Model NLL
    if args.metric in ['all', 'world_model_nll', 'accuracy']:
        if args.data_dir is None:
            data_dir = Path(config['data']['data_dir']) / config['data']['env_name'].replace('/', '_')
        else:
            data_dir = Path(args.data_dir)
        
        print(f"\nLoading test data from {data_dir}")
        
        test_dataset = BabyAIDataset(
            data_dir=str(data_dir),
            split='test',
            max_trajectory_length=config['data']['max_trajectory_length'],
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['data']['batch_size'],
            shuffle=False,
            num_workers=config['data']['num_workers'],
            collate_fn=collate_fn,
        )
        
        print(f"Loaded {len(test_dataset)} test trajectories")
    
    if args.metric in ['all', 'world_model_nll']:
        print("\n=== Computing World Model NLL ===")
        nll = compute_world_model_nll(model, test_loader, device)
        results['world_model_nll'] = nll
        print(f"World Model NLL: {nll:.4f}")
    
    if args.metric in ['all', 'accuracy']:
        print("\n=== Computing Accuracy Metrics ===")
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                observations = batch['observations'].to(device)
                actions = batch['actions'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                outputs = model(
                    observations=observations,
                    actions=actions,
                    attention_mask=attention_mask,
                )
                
                all_predictions.append(outputs)
                all_targets.append({
                    'observations': observations,
                    'actions': actions,
                })
        
        accuracy_metrics = compute_metrics(all_predictions, all_targets)
        results.update(accuracy_metrics)
        
        print("Accuracy Metrics:")
        for k, v in accuracy_metrics.items():
            print(f"  {k}: {v:.4f}")
    
    # Task Success Rate
    if args.metric in ['all', 'task_success']:
        env_name = args.env or config['data']['env_name']
        print(f"\n=== Evaluating Task Success on {env_name} ===")
        
        success_metrics = evaluate_task_success(
            model=model,
            env_name=env_name,
            num_episodes=args.num_episodes,
            device=device,
        )
        
        results['task_success'] = success_metrics
        
        print("Task Success Metrics:")
        for k, v in success_metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
    
    # Partial Observability Robustness
    if args.metric in ['all', 'robustness']:
        env_name = args.env or config['data']['env_name']
        print(f"\n=== Evaluating Robustness on {env_name} ===")
        
        robustness_metrics = evaluate_partial_observability_robustness(
            model=model,
            env_name=env_name,
            num_episodes=args.num_episodes,
            occlusion_prob=args.occlusion_prob,
            device=device,
        )
        
        results['robustness'] = robustness_metrics
        
        print("Robustness Metrics:")
        for k, v in robustness_metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {output_path}")
    else:
        # Save to checkpoint directory
        checkpoint_dir = Path(args.checkpoint).parent.parent
        results_file = checkpoint_dir / 'evaluation_results.json'
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {results_file}")
    
    print("\n=== Evaluation Complete ===")


if __name__ == '__main__':
    main()
