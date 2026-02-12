"""
Evaluation Metrics

Metrics for evaluating world model quality and agent performance.
"""

import torch
import numpy as np
from typing import Dict, List
from sklearn.metrics import accuracy_score


def compute_metrics(predictions: List[Dict], targets: List[Dict]) -> Dict[str, float]:
    """
    Compute evaluation metrics from predictions and targets.
    
    Args:
        predictions: List of prediction dictionaries
        targets: List of target dictionaries
        
    Returns:
        Dictionary of metrics
    """
    # Collect all predictions and targets
    all_obs_pred = []
    all_obs_target = []
    all_act_pred = []
    all_act_target = []
    
    for pred, tgt in zip(predictions, targets):
        if 'observations' in pred:
            obs_logits = pred['observations']  # (batch, T, 7, 7, 3, num_classes)
            obs_pred = obs_logits.argmax(dim=-1)  # (batch, T, 7, 7, 3)
            all_obs_pred.append(obs_pred.cpu().numpy())
            all_obs_target.append(tgt['observations'].cpu().numpy())
        
        if 'actions' in pred:
            act_logits = pred['actions']  # (batch, T, num_actions)
            act_pred = act_logits.argmax(dim=-1)  # (batch, T)
            all_act_pred.append(act_pred.cpu().numpy())
            all_act_target.append(tgt['actions'].cpu().numpy())
    
    metrics = {}
    
    # Observation accuracy
    if all_obs_pred:
        obs_pred = np.concatenate(all_obs_pred, axis=0)
        obs_target = np.concatenate(all_obs_target, axis=0)
        
        # Overall accuracy
        obs_accuracy = (obs_pred == obs_target).mean()
        metrics['observation_accuracy'] = obs_accuracy
        
        # Per-attribute accuracy
        obj_accuracy = (obs_pred[..., 0] == obs_target[..., 0]).mean()
        color_accuracy = (obs_pred[..., 1] == obs_target[..., 1]).mean()
        state_accuracy = (obs_pred[..., 2] == obs_target[..., 2]).mean()
        
        metrics['object_accuracy'] = obj_accuracy
        metrics['color_accuracy'] = color_accuracy
        metrics['state_accuracy'] = state_accuracy
    
    # Action accuracy
    if all_act_pred:
        act_pred = np.concatenate(all_act_pred, axis=0)
        act_target = np.concatenate(all_act_target, axis=0)
        
        act_accuracy = (act_pred == act_target).mean()
        metrics['action_accuracy'] = act_accuracy
    
    return metrics


def compute_world_model_nll(
    model,
    dataloader,
    device: str = 'cuda',
) -> float:
    """
    Compute world model negative log-likelihood.
    
    Args:
        model: Trained model
        dataloader: Data loader
        device: Device to use
        
    Returns:
        Average NLL
    """
    model.eval()
    total_nll = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            observations = batch['observations'].to(device)
            actions = batch['actions'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass without masking (full context)
            outputs = model(
                observations=observations,
                actions=actions,
                attention_mask=attention_mask,
            )
            
            # Compute NLL
            if 'observations' in outputs:
                obs_logits = outputs['observations']
                obs_targets = observations
                
                # Reshape and compute cross-entropy
                batch_size, T, h, w, c, num_classes = obs_logits.shape
                obs_logits_flat = obs_logits.reshape(-1, num_classes)
                obs_targets_flat = obs_targets.reshape(-1)
                
                # Only compute on valid positions
                valid_mask = attention_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                valid_mask = valid_mask.expand(-1, -1, h, w, c).reshape(-1)
                
                obs_logits_valid = obs_logits_flat[valid_mask.bool()]
                obs_targets_valid = obs_targets_flat[valid_mask.bool()]
                
                nll = torch.nn.functional.cross_entropy(
                    obs_logits_valid,
                    obs_targets_valid,
                    reduction='mean'
                )
                
                total_nll += nll.item()
                num_batches += 1
    
    return total_nll / num_batches if num_batches > 0 else 0.0


def evaluate_partial_observability_robustness(
    model,
    env_name: str,
    num_episodes: int = 100,
    occlusion_prob: float = 0.3,
    device: str = 'cuda',
) -> Dict[str, float]:
    """
    Evaluate model robustness under partial observability.
    
    Tests how well the model performs when parts of the observation are occluded.
    
    Args:
        model: Trained model
        env_name: Environment name
        num_episodes: Number of test episodes
        occlusion_prob: Probability of occluding each cell
        device: Device to use
        
    Returns:
        Dictionary with robustness metrics
    """
    import gymnasium as gym
    
    model.eval()
    env = gym.make(env_name, render_mode=None)
    
    successes = 0
    total_reward = 0
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # Apply random occlusion to observation
            obs_tensor = torch.from_numpy(obs['image']).unsqueeze(0).to(device)
            
            # Randomly occlude cells
            occlusion_mask = torch.rand_like(obs_tensor.float()) < occlusion_prob
            obs_occluded = obs_tensor.clone()
            obs_occluded[occlusion_mask] = 0  # Set occluded cells to empty
            
            # Get action from model
            with torch.no_grad():
                # Create dummy action tensor for forward pass
                dummy_actions = torch.zeros(1, 1, dtype=torch.long, device=device)
                attention_mask = torch.ones(1, 1, device=device)
                
                outputs = model(
                    observations=obs_occluded.unsqueeze(1),
                    actions=dummy_actions,
                    attention_mask=attention_mask,
                )
                
                # Get predicted action
                action_logits = outputs['actions'][:, -1]  # Last timestep
                action = action_logits.argmax(dim=-1).item()
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        
        total_reward += episode_reward
        if episode_reward > 0:
            successes += 1
    
    env.close()
    
    return {
        'success_rate': successes / num_episodes,
        'avg_reward': total_reward / num_episodes,
        'occlusion_prob': occlusion_prob,
    }


def evaluate_task_success(
    model,
    env_name: str,
    num_episodes: int = 100,
    device: str = 'cuda',
) -> Dict[str, float]:
    """
    Evaluate task success rate in the environment.
    
    Args:
        model: Trained model
        env_name: Environment name
        num_episodes: Number of test episodes
        device: Device to use
        
    Returns:
        Dictionary with success metrics
    """
    import gymnasium as gym
    
    model.eval()
    env = gym.make(env_name, render_mode=None)
    
    successes = 0
    total_reward = 0
    episode_lengths = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        max_steps = 100
        
        while not done and steps < max_steps:
            # Get observation
            obs_tensor = torch.from_numpy(obs['image']).unsqueeze(0).to(device)
            
            # Get action from model
            with torch.no_grad():
                dummy_actions = torch.zeros(1, 1, dtype=torch.long, device=device)
                attention_mask = torch.ones(1, 1, device=device)
                
                outputs = model(
                    observations=obs_tensor.unsqueeze(1),
                    actions=dummy_actions,
                    attention_mask=attention_mask,
                )
                
                action_logits = outputs['actions'][:, -1]
                action = action_logits.argmax(dim=-1).item()
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            steps += 1
        
        total_reward += episode_reward
        episode_lengths.append(steps)
        if episode_reward > 0:
            successes += 1
    
    env.close()
    
    return {
        'success_rate': successes / num_episodes,
        'avg_reward': total_reward / num_episodes,
        'avg_episode_length': np.mean(episode_lengths),
        'std_episode_length': np.std(episode_lengths),
    }
