#!/usr/bin/env python3
"""
Generate Trajectories from MiniGrid/BabyAI Environments

This script collects expert or random trajectories from MiniGrid/BabyAI environments
and saves them in the format expected by the training pipeline.
"""

import gymnasium as gym
import minigrid
import numpy as np
import json
from pathlib import Path
import argparse
from tqdm import tqdm
from typing import List, Dict


def generate_random_trajectory(env, max_steps: int = 100) -> Dict:
    """
    Generate a random trajectory by taking random actions.
    
    Args:
        env: Gymnasium environment
        max_steps: Maximum number of steps
        
    Returns:
        Dictionary with trajectory data
    """
    observations = []
    actions = []
    rewards = []
    dones = []
    
    obs, info = env.reset()
    mission = info.get('mission', '')
    
    for step in range(max_steps):
        # Store observation
        observations.append(obs['image'].tolist())  # 7x7x3 grid
        
        # Take random action
        action = env.action_space.sample()
        actions.append(int(action))
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        rewards.append(float(reward))
        dones.append(int(done))
        
        if done:
            break
    
    return {
        'observations': observations,
        'actions': actions,
        'rewards': rewards,
        'dones': dones,
        'mission': mission,
        'success': rewards[-1] > 0 if len(rewards) > 0 else False,
        'length': len(actions),
    }


def generate_expert_trajectory(env, max_steps: int = 100) -> Dict:
    """
    Generate an expert trajectory using the optimal policy if available.
    
    Note: This requires the environment to have an optimal policy.
    For now, this is a placeholder that uses random actions.
    In practice, you would use BabyAI's bot or A* planning.
    
    Args:
        env: Gymnasium environment
        max_steps: Maximum number of steps
        
    Returns:
        Dictionary with trajectory data
    """
    # TODO: Implement expert policy (e.g., using BabyAI bot)
    # For now, use random policy as placeholder
    return generate_random_trajectory(env, max_steps)


def collect_trajectories(
    env_name: str,
    num_episodes: int,
    max_steps: int = 100,
    policy: str = 'random',
    output_dir: str = 'data/raw',
    split: str = 'train',
) -> List[Dict]:
    """
    Collect multiple trajectories from an environment.
    
    Args:
        env_name: Name of the MiniGrid/BabyAI environment
        num_episodes: Number of episodes to collect
        max_steps: Maximum steps per episode
        policy: 'random' or 'expert'
        output_dir: Directory to save trajectories
        split: Data split ('train', 'val', 'test')
        
    Returns:
        List of trajectory dictionaries
    """
    # Create environment
    env = gym.make(env_name, render_mode=None)
    
    # Create output directory
    output_path = Path(output_dir) / env_name.replace('/', '_') / split
    output_path.mkdir(parents=True, exist_ok=True)
    
    trajectories = []
    successful_episodes = 0
    
    print(f"Collecting {num_episodes} trajectories from {env_name}...")
    print(f"Policy: {policy}, Max steps: {max_steps}")
    print(f"Saving to: {output_path}")
    
    for episode in tqdm(range(num_episodes)):
        # Generate trajectory
        if policy == 'expert':
            trajectory = generate_expert_trajectory(env, max_steps)
        else:
            trajectory = generate_random_trajectory(env, max_steps)
        
        # Save trajectory
        trajectory_file = output_path / f'trajectory_{episode:06d}.json'
        with open(trajectory_file, 'w') as f:
            json.dump(trajectory, f, indent=2)
        
        trajectories.append(trajectory)
        
        if trajectory['success']:
            successful_episodes += 1
    
    env.close()
    
    # Print statistics
    success_rate = successful_episodes / num_episodes
    avg_length = np.mean([t['length'] for t in trajectories])
    
    print(f"\nCollection complete!")
    print(f"Success rate: {success_rate:.2%} ({successful_episodes}/{num_episodes})")
    print(f"Average trajectory length: {avg_length:.1f}")
    print(f"Trajectories saved to: {output_path}")
    
    # Save metadata
    metadata = {
        'env_name': env_name,
        'num_episodes': num_episodes,
        'policy': policy,
        'max_steps': max_steps,
        'success_rate': success_rate,
        'avg_length': avg_length,
        'split': split,
    }
    
    metadata_file = output_path.parent / f'{split}_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return trajectories


def main():
    parser = argparse.ArgumentParser(description='Generate MiniGrid/BabyAI trajectories')
    
    parser.add_argument('--env', type=str, default='BabyAI-GoToRedBall-v0',
                        help='Environment name')
    parser.add_argument('--num_episodes', type=int, default=1000,
                        help='Number of episodes to collect')
    parser.add_argument('--max_steps', type=int, default=100,
                        help='Maximum steps per episode')
    parser.add_argument('--policy', type=str, default='random',
                        choices=['random', 'expert'],
                        help='Policy to use for collection')
    parser.add_argument('--output_dir', type=str, default='data/raw',
                        help='Output directory')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Fraction of data for training')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Fraction of data for validation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    np.random.seed(args.seed)
    
    # Calculate split sizes
    total_episodes = args.num_episodes
    train_episodes = int(total_episodes * args.train_ratio)
    val_episodes = int(total_episodes * args.val_ratio)
    test_episodes = total_episodes - train_episodes - val_episodes
    
    # Collect train split
    print("\n=== Collecting Train Split ===")
    collect_trajectories(
        env_name=args.env,
        num_episodes=train_episodes,
        max_steps=args.max_steps,
        policy=args.policy,
        output_dir=args.output_dir,
        split='train',
    )
    
    # Collect val split
    print("\n=== Collecting Val Split ===")
    collect_trajectories(
        env_name=args.env,
        num_episodes=val_episodes,
        max_steps=args.max_steps,
        policy=args.policy,
        output_dir=args.output_dir,
        split='val',
    )
    
    # Collect test split
    print("\n=== Collecting Test Split ===")
    collect_trajectories(
        env_name=args.env,
        num_episodes=test_episodes,
        max_steps=args.max_steps,
        policy=args.policy,
        output_dir=args.output_dir,
        split='test',
    )
    
    print("\n=== All Done! ===")


if __name__ == '__main__':
    main()
