import os
import torch
import numpy as np
from typing import Dict, List, Any
from tqdm import tqdm
import alfworld
import alfworld.agents.environment
from .metrics import compute_observation_masked_nll
from ..model.inference import generate_action, corrupt_observation

def evaluate_alfworld(
    model, tokenizer, config,
    split: str = "eval_out_of_distribution",
    rho: float = 0.0,
    n_episodes: int = 50,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Runs the AOMT agent on ALFWorld.
    """
    os.environ['ALFWORLD_DATA'] = os.path.expanduser('~/.alfworld')
    
    # Load ALFWorld config
    config_path = os.path.join(os.path.dirname(__file__), "alfworld_config.yaml")
    if not os.path.exists(config_path):
        import alfworld.agents.modules.generic as common
        alf_cfg = common.load_config(os.path.join(os.path.dirname(alfworld.__file__), "configs/base_config.yaml"))
    else:
        with open(config_path, "r") as f:
            import yaml
            alf_cfg = yaml.safe_load(f)

    env_class = alfworld.agents.environment.AlfredTWEnv
    env = env_class(alf_cfg, split=split)
    env.seed(seed)
    
    rng = np.random.default_rng(seed)
    successes = 0
    
    print(f"Evaluating ALFWorld | split={split} | rho={rho}")
    
    for _ in tqdm(range(n_episodes), desc="ALFWorld Episodes"):
        obs, info = env.reset()
        done = False
        step = 0
        # history_parts is a list of strings [O0, A0, O1, A1, ..., Ot]
        history_parts = []
        
        # Initial observation corruption
        obs_str = corrupt_observation(obs[0], tokenizer, rho, rng) if rho > 0 else obs[0]
        history_parts.append(obs_str)
        
        while not done and step < 50:
            # All methods now use identical inference
            action = generate_action(
                model, tokenizer, history_parts, 
                gen_length=config.get("max_new_tokens", 256),
                steps=config.get("diffusion_steps", 32),
                temperature=config.get("temperature", 0.0)
            )
            
            # ALFWorld env expects a list of actions
            obs, reward, done, info = env.step([action])
            
            obs_str = obs[0]
            if rho > 0:
                obs_str = corrupt_observation(obs_str, tokenizer, rho, rng)
            
            history_parts.append(action)
            history_parts.append(obs_str)
            
            if reward[0] > 0:
                successes += 1
                done = True
            
            step += 1
            
    env.close()
    
    return {
        "success_rate": (successes / n_episodes) * 100,
        "n_episodes": n_episodes,
        "n_success": successes,
    }
