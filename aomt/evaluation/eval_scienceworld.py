import torch
import numpy as np
from typing import Dict, List, Any, Optional
from tqdm import tqdm
from scienceworld import ScienceWorldEnv
from ..model.inference import generate_action

def evaluate_scienceworld(
    model, tokenizer, config,
    task_ids: Optional[List[int]] = None,
    n_episodes_per_task: int = 1,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Evaluates the AOMT agent on ScienceWorld tasks.
    """
    env = ScienceWorldEnv()
    if task_ids is None:
        task_ids = list(range(30))
        
    all_scores = []
    per_task_results = {}
    
    print(f"Evaluating ScienceWorld | tasks={len(task_ids)}")
    
    for task_id in tqdm(task_ids, desc="ScienceWorld Tasks"):
        task_scores = []
        task_name = env.getTaskNames()[task_id]
        
        for ep_idx in range(n_episodes_per_task):
            env.load(task_name, ep_idx)
            obs, info = env.reset()
            done = False
            step = 0
            history_parts = [obs]
            
            while not done and step < 100:
                action = generate_action(
                    model, tokenizer, history_parts,
                    gen_length=config.get("max_new_tokens", 256),
                    steps=config.get("diffusion_steps", 32),
                    temperature=config.get("temperature", 0.0)
                )
                
                obs, reward, done, info = env.step(action)
                history_parts.append(action)
                history_parts.append(obs)
                
                step += 1
            
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
