"""
MiniGrid/BabyAI Dataset Loader

Loads and processes trajectories from MiniGrid/BabyAI environments.
Observation format: 7x7x3 grid (partially observable)
Each cell has 3 attributes: [object_type, color, state]
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import gymnasium as gym


class MiniGridDataset(Dataset):
    """
    Dataset for MiniGrid/BabyAI trajectories.
    
    Each trajectory contains:
    - observations: List of 7x7x3 grids
    - actions: List of action indices
    - missions: Natural language mission description
    - rewards: List of rewards
    - dones: List of done flags
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        max_trajectory_length: int = 128,
        transform=None,
    ):
        """
        Args:
            data_dir: Directory containing processed trajectory files
            split: One of 'train', 'val', 'test'
            max_trajectory_length: Maximum trajectory length (truncate/pad)
            transform: Optional transform to apply to observations
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_trajectory_length = max_trajectory_length
        self.transform = transform
        
        # Load trajectory files
        self.trajectory_files = sorted(
            list((self.data_dir / split).glob("*.json"))
        )
        
        if len(self.trajectory_files) == 0:
            raise ValueError(f"No trajectory files found in {self.data_dir / split}")
        
        print(f"Loaded {len(self.trajectory_files)} trajectories from {split} split")
    
    def __len__(self) -> int:
        return len(self.trajectory_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a single trajectory.
        
        Returns:
            Dictionary with keys:
            - observations: (seq_len, 7, 7, 3) tensor
            - actions: (seq_len,) tensor
            - mission: string
            - rewards: (seq_len,) tensor
            - dones: (seq_len,) tensor
            - attention_mask: (seq_len,) binary mask for valid positions
        """
        # Load trajectory
        with open(self.trajectory_files[idx], 'r') as f:
            trajectory = json.load(f)
        
        # Extract components
        observations = np.array(trajectory['observations'])  # (T, 7, 7, 3)
        actions = np.array(trajectory['actions'])  # (T,)
        mission = trajectory['mission']
        rewards = np.array(trajectory['rewards'])  # (T,)
        dones = np.array(trajectory['dones'])  # (T,)
        
        seq_len = len(actions)
        
        # Truncate or pad to max_trajectory_length
        if seq_len > self.max_trajectory_length:
            observations = observations[:self.max_trajectory_length]
            actions = actions[:self.max_trajectory_length]
            rewards = rewards[:self.max_trajectory_length]
            dones = dones[:self.max_trajectory_length]
            attention_mask = np.ones(self.max_trajectory_length, dtype=np.float32)
        else:
            # Pad
            pad_len = self.max_trajectory_length - seq_len
            observations = np.pad(
                observations,
                ((0, pad_len), (0, 0), (0, 0), (0, 0)),
                mode='constant',
                constant_values=0
            )
            actions = np.pad(actions, (0, pad_len), mode='constant', constant_values=0)
            rewards = np.pad(rewards, (0, pad_len), mode='constant', constant_values=0)
            dones = np.pad(dones, (0, pad_len), mode='constant', constant_values=0)
            
            # Attention mask: 1 for valid, 0 for padding
            attention_mask = np.concatenate([
                np.ones(seq_len, dtype=np.float32),
                np.zeros(pad_len, dtype=np.float32)
            ])
        
        # Apply transform if provided
        if self.transform is not None:
            observations = self.transform(observations)
        
        return {
            'observations': torch.from_numpy(observations).long(),
            'actions': torch.from_numpy(actions).long(),
            'mission': mission,
            'rewards': torch.from_numpy(rewards).float(),
            'dones': torch.from_numpy(dones).float(),
            'attention_mask': torch.from_numpy(attention_mask).float(),
        }


class BabyAIDataset(MiniGridDataset):
    """
    Specialized dataset for BabyAI environments.
    Inherits from MiniGridDataset with BabyAI-specific processing.
    """
    
    # BabyAI object types
    OBJECT_TO_IDX = {
        'unseen': 0,
        'empty': 1,
        'wall': 2,
        'floor': 3,
        'door': 4,
        'key': 5,
        'ball': 6,
        'box': 7,
        'goal': 8,
        'lava': 9,
        'agent': 10,
    }
    
    # BabyAI colors
    COLOR_TO_IDX = {
        'red': 0,
        'green': 1,
        'blue': 2,
        'purple': 3,
        'yellow': 4,
        'grey': 5,
    }
    
    # States for objects
    STATE_TO_IDX = {
        'open': 0,
        'closed': 1,
        'locked': 2,
    }
    
    # Action space
    ACTION_TO_IDX = {
        'left': 0,
        'right': 1,
        'forward': 2,
        'pickup': 3,
        'drop': 4,
        'toggle': 5,
        'done': 6,
    }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Create reverse mappings
        self.idx_to_object = {v: k for k, v in self.OBJECT_TO_IDX.items()}
        self.idx_to_color = {v: k for k, v in self.COLOR_TO_IDX.items()}
        self.idx_to_state = {v: k for k, v in self.STATE_TO_IDX.items()}
        self.idx_to_action = {v: k for k, v in self.ACTION_TO_IDX.items()}
    
    def decode_observation(self, obs: np.ndarray) -> str:
        """
        Decode observation tensor to human-readable string.
        
        Args:
            obs: (7, 7, 3) observation array
            
        Returns:
            String representation of the grid
        """
        lines = []
        for i in range(7):
            row = []
            for j in range(7):
                obj_idx, color_idx, state_idx = obs[i, j]
                obj = self.idx_to_object.get(obj_idx, 'unknown')
                color = self.idx_to_color.get(color_idx, 'unknown')
                state = self.idx_to_state.get(state_idx, 'unknown')
                
                if obj == 'empty':
                    row.append('  .  ')
                elif obj == 'wall':
                    row.append('  #  ')
                elif obj == 'agent':
                    row.append('  @  ')
                else:
                    row.append(f"{obj[0]}{color[0]}{state[0]}")
            lines.append('|'.join(row))
        return '\n'.join(lines)


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for batching trajectories.
    """
    # Stack all tensors
    observations = torch.stack([item['observations'] for item in batch])
    actions = torch.stack([item['actions'] for item in batch])
    rewards = torch.stack([item['rewards'] for item in batch])
    dones = torch.stack([item['dones'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    
    # Keep missions as list of strings
    missions = [item['mission'] for item in batch]
    
    return {
        'observations': observations,
        'actions': actions,
        'missions': missions,
        'rewards': rewards,
        'dones': dones,
        'attention_mask': attention_mask,
    }
