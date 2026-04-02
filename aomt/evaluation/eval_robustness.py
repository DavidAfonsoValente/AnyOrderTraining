import torch
import numpy as np
from typing import Dict
from .eval_alfworld import evaluate_alfworld

def evaluate_robustness(
    model, tokenizer, config,
    rhos: list = [0.0, 0.1, 0.2, 0.3],
    split: str = "eval_in_distribution",
    inference_mode: str = "mode_a",
    seed: int = 42,
) -> Dict:
    """
    Evaluates robustness under observation corruption.
    """
    results = {}
    for rho in rhos:
        print(f"Running robustness test: rho={rho}")
        res = evaluate_alfworld(
            model, tokenizer, config, 
            split=split, 
            inference_mode=inference_mode,
            rho=rho,
            n_episodes=20, # Shorter eval for robustness curve
            seed=seed
        )
        results[rho] = res["success_rate"]
        
    return results
