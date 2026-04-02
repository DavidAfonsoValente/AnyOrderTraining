import torch
import numpy as np
from typing import Dict, Any
from tqdm import tqdm
import sys
import os

# Ensure WebShop is in path
sys.path.append(os.path.join(os.getcwd(), "third_party/WebShop"))

from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
from ..model.inference import generate_next_action_mode_a, generate_next_action_mode_b
from scripts.webshop_server import start_webshop_server

def evaluate_webshop(
    model, tokenizer, config,
    n_sessions: int = 500,
    inference_mode: str = "mode_a",
    planning_horizon: int = 3,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Evaluates the AOMT agent on WebShop.
    """
    # Start the local Flask server
    print("Starting WebShop server...")
    server_proc, port = start_webshop_server()
    
    try:
        # Connect environment to the local server
        env = WebAgentTextEnv(observation_mode="text", server_port=port)
        
        rewards = []
        
        print(f"Evaluating WebShop | sessions={n_sessions} | mode={inference_mode} | port={port}")
        
        for session_idx in tqdm(range(n_sessions), desc="WebShop Sessions"):
            obs, info = env.reset()
            done = False
            step = 0
            history = [obs]
            
            while not done and step < 15: # WebShop sessions are typically short
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
                
                # Standard WebShop commands: search[query], click[button]
                obs, reward, done, info = env.step(action)
                history.append(action)
                history.append(obs)
                
                step += 1
            
            rewards.append(reward * 100) # Scale to [0, 100]
            
        return {
            "reward": np.mean(rewards),
            "n_sessions": n_sessions,
            "server_port": port
        }
        
    finally:
        print("Shutting down WebShop server...")
        server_proc.terminate()
        server_proc.wait()
