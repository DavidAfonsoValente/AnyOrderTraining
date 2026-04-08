import torch
import numpy as np
from typing import Dict, Any
from tqdm import tqdm
import sys
import os

# Ensure WebShop is in path
sys.path.append(os.path.join(os.getcwd(), "third_party/WebShop"))

from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
from ..model.inference import generate_action
from scripts.webshop_server import start_webshop_server

def evaluate_webshop(
    model, tokenizer, config,
    n_sessions: int = 500,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Evaluates the AOMT agent on WebShop.
    """
    print("Starting WebShop server...")
    server_proc, port = start_webshop_server()
    
    try:
        env = WebAgentTextEnv(observation_mode="text", server_port=port)
        rewards = []
        
        print(f"Evaluating WebShop | sessions={n_sessions} | port={port}")
        
        for session_idx in tqdm(range(n_sessions), desc="WebShop Sessions"):
            obs, info = env.reset()
            done = False
            step = 0
            history_parts = [obs]
            
            while not done and step < 15:
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
            
            rewards.append(reward * 100)
            
        return {
            "reward": np.mean(rewards),
            "n_sessions": n_sessions,
            "server_port": port
        }
        
    finally:
        print("Shutting down WebShop server...")
        server_proc.terminate()
        server_proc.wait()
