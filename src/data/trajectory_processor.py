"""
Trajectory Processor

Converts raw trajectories into the format expected by Any-Order Masked Training.
Handles interleaving of observations and actions for masked reconstruction.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ProcessedTrajectory:
    """
    Processed trajectory in interleaved Obs/Act format.
    """
    sequence: List[Dict]  # List of {type: 'obs'/'act', data: tensor, position: int}
    mission: str
    attention_mask: torch.Tensor  # Binary mask for valid positions
    metadata: Dict


class TrajectoryProcessor:
    """
    Processes trajectories into interleaved Obs/Act sequences.
    
    Format: Obs_0 → Act_0 → Obs_1 → Act_1 → ... → Obs_T → Act_T
    """
    
    def __init__(
        self,
        max_length: int = 256,  # Max sequence length (obs + acts)
        tokenize_obs: bool = True,  # Whether to tokenize observations
        tokenize_act: bool = True,  # Whether to tokenize actions
    ):
        self.max_length = max_length
        self.tokenize_obs = tokenize_obs
        self.tokenize_act = tokenize_act
    
    def process(
        self,
        observations: torch.Tensor,  # (T, 7, 7, 3)
        actions: torch.Tensor,  # (T,)
        mission: str,
        attention_mask: torch.Tensor,  # (T,)
    ) -> ProcessedTrajectory:
        """
        Process a trajectory into interleaved format.
        
        Returns:
            ProcessedTrajectory with interleaved obs/act sequence
        """
        T = len(actions)
        valid_length = attention_mask.sum().int().item()
        
        # Create interleaved sequence
        sequence = []
        
        for t in range(valid_length):
            # Add observation at time t
            obs_data = observations[t]  # (7, 7, 3)
            
            if self.tokenize_obs:
                # Flatten observation into tokens
                # Each cell becomes one token with 3 attributes
                obs_tokens = self._tokenize_observation(obs_data)
            else:
                obs_tokens = obs_data
            
            sequence.append({
                'type': 'observation',
                'data': obs_tokens,
                'position': 2 * t,  # Even positions for observations
                'timestep': t,
            })
            
            # Add action at time t
            act_data = actions[t]  # scalar action index
            
            if self.tokenize_act:
                act_tokens = self._tokenize_action(act_data)
            else:
                act_tokens = act_data
            
            sequence.append({
                'type': 'action',
                'data': act_tokens,
                'position': 2 * t + 1,  # Odd positions for actions
                'timestep': t,
            })
        
        # Create attention mask for interleaved sequence
        seq_length = len(sequence)
        seq_attention_mask = torch.ones(seq_length)
        
        # Truncate if necessary
        if seq_length > self.max_length:
            sequence = sequence[:self.max_length]
            seq_attention_mask = seq_attention_mask[:self.max_length]
        
        metadata = {
            'original_length': T,
            'valid_length': valid_length,
            'sequence_length': len(sequence),
        }
        
        return ProcessedTrajectory(
            sequence=sequence,
            mission=mission,
            attention_mask=seq_attention_mask,
            metadata=metadata,
        )
    
    def _tokenize_observation(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Tokenize a 7x7x3 observation into a sequence of cell tokens.
        
        Each cell (obj, color, state) becomes one token position.
        
        Args:
            obs: (7, 7, 3) observation tensor
            
        Returns:
            (49, 3) tensor of cell tokens (flattened grid)
        """
        # Flatten 7x7 grid into 49 cells
        # Each cell has 3 attributes: [object, color, state]
        return obs.reshape(-1, 3)  # (49, 3)
    
    def _tokenize_action(self, action: torch.Tensor) -> torch.Tensor:
        """
        Tokenize an action.
        
        Args:
            action: Scalar action index
            
        Returns:
            Action tensor (kept as scalar for simplicity)
        """
        return action.unsqueeze(0) if action.dim() == 0 else action
    
    def detokenize_observation(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Convert tokenized observation back to 7x7x3 grid.
        
        Args:
            tokens: (49, 3) cell tokens
            
        Returns:
            (7, 7, 3) observation tensor
        """
        return tokens.reshape(7, 7, 3)
    
    def detokenize_action(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Convert tokenized action back to scalar.
        
        Args:
            tokens: Action token(s)
            
        Returns:
            Scalar action index
        """
        if tokens.dim() > 0:
            return tokens[0] if len(tokens) == 1 else tokens
        return tokens


def create_interleaved_sequence(
    observations: torch.Tensor,
    actions: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create interleaved observation-action sequence.
    
    Args:
        observations: (batch, T, 7, 7, 3)
        actions: (batch, T)
        attention_mask: (batch, T)
        
    Returns:
        Tuple of:
        - interleaved_sequence: (batch, 2*T, ...) 
        - sequence_type: (batch, 2*T) with 0=obs, 1=act
        - interleaved_mask: (batch, 2*T)
    """
    batch_size, T = actions.shape
    
    # Flatten observations to (batch, T, 49, 3)
    obs_flat = observations.reshape(batch_size, T, -1, 3)
    
    # Expand actions to (batch, T, 1, 1) for concatenation
    act_expanded = actions.unsqueeze(-1).unsqueeze(-1)
    
    # Create type markers: 0 for obs, 1 for act
    obs_types = torch.zeros(batch_size, T, dtype=torch.long, device=observations.device)
    act_types = torch.ones(batch_size, T, dtype=torch.long, device=actions.device)
    
    # Interleave: obs_0, act_0, obs_1, act_1, ...
    interleaved_data = []
    interleaved_types = []
    interleaved_masks = []
    
    for t in range(T):
        interleaved_data.append(obs_flat[:, t])  # (batch, 49, 3)
        interleaved_types.append(obs_types[:, t])  # (batch,)
        interleaved_masks.append(attention_mask[:, t])  # (batch,)
        
        interleaved_data.append(act_expanded[:, t])  # (batch, 1, 1)
        interleaved_types.append(act_types[:, t])  # (batch,)
        interleaved_masks.append(attention_mask[:, t])  # (batch,)
    
    # Stack into sequences
    # Note: This creates a list structure; actual implementation may need refinement
    sequence_type = torch.stack(interleaved_types, dim=1)  # (batch, 2*T)
    interleaved_mask = torch.stack(interleaved_masks, dim=1)  # (batch, 2*T)
    
    return interleaved_data, sequence_type, interleaved_mask
