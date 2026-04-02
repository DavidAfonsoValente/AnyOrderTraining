import torch
import numpy as np
from typing import Dict, List
from scienceworld import ScienceWorldEnv
from ..model.inference import generate_next_action_mode_a, generate_next_action_mode_b

def evaluate_scienceworld(
    model, tokenizer, config,
    task_ids: List[int] = None,
    inference_mode: str = "mode_a",
    planning_horizon: int = 3,
    n_episodes_per_task: int = 5,
    seed: int = 42,
) -> Dict:
    """
    Evaluates on ScienceWorld tasks.
    """
    env = ScienceWorldEnv()
    if task_ids is None:
        task_ids = list(range(30)) # All tasks
        
    scores = []
    
    print(f"Evaluating ScienceWorld | tasks={len(task_ids)} | mode={inference_mode}")
    
    # Placeholder for actual loop
    # For each task, for each episode:
    #   env.load(taskName, taskIdx)
    #   env.reset()
    #   ... loop ...
    
    env.close()
    
    return {
        "all_score": 0.0,
        "per_task_scores": {}
    }
