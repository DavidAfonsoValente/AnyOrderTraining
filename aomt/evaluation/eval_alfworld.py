import os
import torch
import numpy as np
from typing import Dict, List, Optional
from tqdm import tqdm
import alfworld
import alfworld.agents.environment
from .metrics import compute_observation_masked_nll
from ..model.inference import generate_next_action_mode_a, generate_next_action_mode_b, corrupt_observation

def evaluate_alfworld(
    model, tokenizer, config,
    split: str = "eval_out_of_distribution",
    inference_mode: str = "mode_a",
    planning_horizon: int = 3,
    rho: float = 0.0,
    n_episodes: int = 50,
    seed: int = 42,
) -> Dict:
    """
    Runs the AOMT agent on ALFWorld.
    """
    # Load AlfWorld environment
    # Note: AlfWorld requires a config file or env var
    os.environ['ALFWORLD_DATA'] = os.path.expanduser('~/.alfworld')
    
    # This is a simplified version of AlfWorld loop
    # In practice, we'd use alfworld.agents.environment.AlfredTWEnv
    
    rng = np.random.default_rng(seed)
    
    successes = 0
    lengths = []
    
    # Mocking the actual environment loop for now as we can't run it easily here
    # without full setup and game files. 
    # The actual implementation would follow the Spec logic.
    
    print(f"Evaluating ALFWorld | split={split} | mode={inference_mode} | rho={rho}")
    
    # Placeholder for actual loop
    for i in range(n_episodes):
        # success = run_single_episode(...)
        pass
        
    return {
        "success_rate": 0.0, # Placeholder
        "n_episodes": n_episodes,
        "n_success": 0,
        "mean_episode_length": 0.0,
    }
