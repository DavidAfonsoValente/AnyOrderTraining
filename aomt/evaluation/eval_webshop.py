import torch
import numpy as np
from typing import Dict
from ..model.inference import generate_next_action_mode_a, generate_next_action_mode_b

def evaluate_webshop(
    model, tokenizer, config,
    n_sessions: int = 500,
    inference_mode: str = "mode_a",
    planning_horizon: int = 3,
    seed: int = 42,
) -> Dict:
    """
    Evaluates on WebShop.
    """
    # Start server logic here or assumes already running
    
    print(f"Evaluating WebShop | sessions={n_sessions} | mode={inference_mode}")
    
    # Placeholder for actual loop
    
    return {
        "reward": 0.0,
        "n_sessions": n_sessions
    }
