"""
eval/task_eval.py
Final evaluation suite for ALFWorld, ScienceWorld, and WebShop.
Grounded in NeurIPS paper specifications.
- Identical chat-template inference for all methods.
- Fixed hyperparameters: temp=0.0, steps=32, gen_length=256.
- Per-category tracking for ALFWorld.
- All 30 ScienceWorld tasks with continuous scoring [0,1].
- Token-level corruption for robustness testing.
"""

import argparse
import json
import os
import sys
import torch
import numpy as np
import yaml
import random
from tqdm import tqdm
from transformers import AutoTokenizer
from aomt.inference import load_model_for_eval, generate_action, extract_action_from_react

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def corrupt_observation(obs: str, rate: float, tokenizer, rng: np.random.Generator) -> str:
    """Randomly replace fraction rho of tokens with random vocab tokens."""
    if rate <= 0: return obs
    ids = tokenizer.encode(obs, add_special_tokens=False)
    if not ids: return obs
    n_corrupt = max(1, int(rate * len(ids)))
    corrupt_positions = rng.choice(len(ids), size=n_corrupt, replace=False)
    ids = list(ids)
    vocab_size = tokenizer.vocab_size
    for pos in corrupt_positions:
        rand_id = int(rng.integers(1000, vocab_size))
        ids[pos] = rand_id
    return tokenizer.decode(ids, skip_special_tokens=True)

def get_alfworld_category(infos: dict) -> str:
    gamefile = infos.get('extra.gamefile', [''])[0]
    if 'pick_and_place_simple' in gamefile: return 'Pick & Place'
    if 'pick_clean_then_place' in gamefile: return 'Clean & Place'
    if 'pick_heat_then_place' in gamefile: return 'Heat & Place'
    if 'pick_cool_then_place' in gamefile: return 'Cool & Place'
    if 'look_at_obj_in_light' in gamefile: return 'Examine in Light'
    if 'pick_two_obj_and_place' in gamefile: return 'Pick Two & Place'
    return 'unknown'

# -----------------------------------------------------------------------------
# ALFWorld
# -----------------------------------------------------------------------------

def evaluate_alfworld(model, tokenizer, config_path, split="unseen", n_episodes=50, 
                      obs_corruption_rate=0.0, **kwargs):
    from alfworld.agents.environment import get_environment
    with open(config_path) as f: config = yaml.safe_load(f)
    
    split_map = {"seen": "eval_in_distribution", "unseen": "eval_out_of_distribution"}
    train_eval = split_map.get(split, "eval_out_of_distribution")
    
    env = get_environment(config.get("env", {}).get("type", "AlfredTWEnv"))(config, train_eval=train_eval)
    gym_env = env.init_env(batch_size=1)
    
    rng = np.random.default_rng(42)
    successes = []
    cat_results = {}

    for ep in tqdm(range(min(n_episodes, env.num_games)), desc="ALFWorld"):
        obs, infos = gym_env.reset()
        obs = str(obs[0])
        cat = get_alfworld_category(infos)
        if obs_corruption_rate > 0: obs = corrupt_observation(obs, obs_corruption_rate, tokenizer, rng)
        
        history = [obs]
        done = False
        final_reward = 0.0
        for _ in range(50): # Max 50 steps
            prompt = [{"role": "user", "content": "\n".join(history)}]
            raw = generate_action(model, tokenizer, prompt, **kwargs)
            action = extract_action_from_react(raw)
            
            obs_list, rewards, dones, _ = gym_env.step([action])
            obs = str(obs_list[0])
            if obs_corruption_rate > 0: obs = corrupt_observation(obs, obs_corruption_rate, tokenizer, rng)
            
            history.extend([raw, obs])
            if dones[0]:
                final_reward = rewards[0]
                break
        
        success = 1.0 if final_reward > 0 else 0.0
        successes.append(success)
        cat_results.setdefault(cat, []).append(success)

    gym_env.close()
    return {
        "success_rate": float(np.mean(successes)) * 100,
        "category_results": {c: float(np.mean(r)) * 100 for c, r in cat_results.items()}
    }

# -----------------------------------------------------------------------------
# ScienceWorld
# -----------------------------------------------------------------------------

SCIWORLD_TASKS = [
    "boil", "change-state-of-matter", "melt", "combustion", "grow-plant", "freeze",
    "electrical-conductivity", "magnetism", "circuits", "light-shadow", "sound-waves",
    "mix-solutions", "separate-mixtures", "chemical-reactions", "dissolving", "ph-test",
    "measure-temperature", "measure-mass", "measure-volume", "measure-length", "use-ph-meter",
    "find-living", "find-non-living", "identify-animals", "identify-plants", "life-cycles",
    "photosynthesis", "food-chains", "ecosystems", "heat-transfer"
]

def evaluate_scienceworld(model, tokenizer, split="unseen", n_episodes=50, **kwargs):
    from scienceworld import ScienceWorldEnv
    env = ScienceWorldEnv("", "", envStepLimit=100)
    
    scores = []
    ep = 0
    for task_name in tqdm(SCIWORLD_TASKS, desc="SciWorld Tasks"):
        if ep >= n_episodes: break
        env.load(task_name, 0)
        variations = env.get_variations_test() if split == "unseen" else env.get_variations_train()
        
        for var_idx in variations:
            if ep >= n_episodes: break
            env.load(task_name, var_idx)
            obs, info = env.reset()
            history = [str(obs)]
            done = False
            for _ in range(100):
                prompt = [{"role": "user", "content": "\n".join(history)}]
                raw = generate_action(model, tokenizer, prompt, **kwargs)
                action = extract_action_from_react(raw)
                obs, reward, done, info = env.step(action)
                history.extend([raw, str(obs)])
                if done: break
            scores.append(info.get('score', 0.0) / 100.0)
            ep += 1
            
    env.close()
    return {"avg_score": float(np.mean(scores)) * 100 if scores else 0.0}

# -----------------------------------------------------------------------------
# WebShop
# -----------------------------------------------------------------------------

def evaluate_webshop(model, tokenizer, split="test", n_episodes=50, **kwargs):
    import requests
    # Ensure server is reachable (assume user starts it per README)
    try: requests.get("http://localhost:3000", timeout=2)
    except: raise RuntimeError("WebShop server not found at localhost:3000")

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'WebShop'))
    try: from web_agent_site.envs import WebAgentTextEnv
    except ImportError: from webshop.env import WebAgentTextEnv
    
    env = WebAgentTextEnv(observation_mode="text", split=split)
    rewards = []
    for idx in tqdm(range(n_episodes), desc="WebShop"):
        obs, info = env.reset(idx)
        history = [str(obs)]
        done = False
        for _ in range(10): # Max 10 steps for WebShop
            prompt = [{"role": "user", "content": "\n".join(history)}]
            raw = generate_action(model, tokenizer, prompt, **kwargs)
            action = extract_action_from_react(raw)
            obs, reward, done, info = env.step(action)
            history.extend([raw, str(obs)])
            if done: break
        rewards.append(reward)
    return {"avg_reward": float(np.mean(rewards)) * 100 if rewards else 0.0}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--benchmark", type=str, required=True, choices=["alfworld", "scienceworld", "webshop"])
    parser.add_argument("--split", type=str, default="unseen")
    parser.add_argument("--n_episodes", type=int, default=50)
    parser.add_argument("--obs_corruption_rate", type=float, default=0.0)
    parser.add_argument("--output_json", type=str, required=True)
    args = parser.parse_args()

    tok_path = args.tokenizer if args.tokenizer else args.checkpoint_dir
    model, tokenizer, _ = load_model_for_eval(args.checkpoint_dir, tok_path)

    # Fixed paper hyperparameters
    params = {"steps": 32, "gen_length": 256, "temperature": 0.0}

    if args.benchmark == "alfworld":
        conf = os.environ.get("ALFWORLD_CONFIG", "aomt/configs/alfworld_config.yaml")
        res = evaluate_alfworld(model, tokenizer, conf, args.split, args.n_episodes, 
                                args.obs_corruption_rate, **params)
    elif args.benchmark == "scienceworld":
        res = evaluate_scienceworld(model, tokenizer, args.split, args.n_episodes, **params)
    elif args.benchmark == "webshop":
        res = evaluate_webshop(model, tokenizer, args.split, args.n_episodes, **params)

    res.update({"method": args.method, "benchmark": args.benchmark, "split": args.split})
    with open(args.output_json, "w") as f: json.dump(res, f, indent=4)
    print(f"\nFinal Result: {res}")

if __name__ == "__main__": main()
