import os
import torch
import numpy as np
from typing import Dict, List, Any
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
) -> Dict[str, Any]:
    """
    Runs the AOMT agent on ALFWorld.
    """
    # AlfWorld expects a config file. We'll use the standard one from the library.
    # The split mapping matches the ETO dataset expectations.
    os.environ['ALFWORLD_DATA'] = os.path.expanduser('~/.alfworld')
    
    # Load ALFWorld config
    config_path = os.path.join(os.path.dirname(__file__), "alfworld_config.yaml")
    # Fallback to default if not exists
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
    results_by_type = {}
    
    print(f"Evaluating ALFWorld | split={split} | mode={inference_mode} | rho={rho}")
    
    for _ in tqdm(range(n_episodes), desc="ALFWorld Episodes"):
        obs, info = env.reset()
        done = False
        step = 0
        history = []
        
        # Initial observation corruption
        obs = corrupt_observation(obs[0], tokenizer, rho, rng) if rho > 0 else obs[0]
        history.append(obs)
        
        while not done and step < 50:
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
            
            # ALFWorld env expects a list of actions
            obs, reward, done, info = env.step([action])
            
            # Extract and corrupt next observation
            obs_str = obs[0]
            if rho > 0:
                obs_str = corrupt_observation(obs_str, tokenizer, rho, rng)
            
            history.append(action)
            history.append(obs_str)
            
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
