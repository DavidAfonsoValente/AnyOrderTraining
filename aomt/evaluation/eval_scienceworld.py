import torch
import numpy as np
from typing import Dict, List, Any, Optional
from tqdm import tqdm
from scienceworld import ScienceWorldEnv
from ..model.inference import generate_next_action_mode_a, generate_next_action_mode_b

def evaluate_scienceworld(
    model, tokenizer, config,
    task_ids: Optional[List[int]] = None,
    inference_mode: str = "mode_a",
    planning_horizon: int = 3,
    n_episodes_per_task: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Evaluates the AOMT agent on ScienceWorld tasks.
    """
    env = ScienceWorldEnv()
    if task_ids is None:
        # Evaluate on all 30 tasks by default
        task_ids = list(range(30))
        
    all_scores = []
    per_task_results = {}
    
    print(f"Evaluating ScienceWorld | tasks={len(task_ids)} | mode={inference_mode}")
    
    for task_id in tqdm(task_ids, desc="ScienceWorld Tasks"):
        task_scores = []
        task_name = env.getTaskNames()[task_id]
        
        for ep_idx in range(n_episodes_per_task):
            env.load(task_name, ep_idx)
            obs, info = env.reset()
            done = False
            step = 0
            history = [obs]
            
            while not done and step < 100: # ScienceWorld tasks can be long
                if inference_mode == "mode_a":
                    action = generate_next_action_mode_a(model, tokenizer, history, device=model.device)
                else:
                    action = generate_next_action_mode_b(
                        model, tokenizer, history, 
                        method="aomt_mixed", 
                        planning_horizon=planning_horizon,
                        median_action_tokens=config.get("median_action_tokens", 33),
                        median_obs_tokens=config.get("median_obs_tokens", 17),
                        device=model.device
                    )
                
                obs, reward, done, info = env.step(action)
                history.append(action)
                history.append(obs)
                
                step += 1
            
            # Normalize score to [0, 100]
            norm_score = env.getScore() * 100
            task_scores.append(norm_score)
            
        avg_task_score = np.mean(task_scores)
        per_task_results[task_name] = avg_task_score
        all_scores.append(avg_task_score)
        
    env.close()
    
    return {
        "all_score": np.mean(all_scores),
        "per_task_scores": per_task_results,
        "n_tasks": len(task_ids)
    }
