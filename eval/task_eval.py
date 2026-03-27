# eval/task_eval.py

import torch
import yaml
import json
import os
import argparse
import random
from tqdm import tqdm
from aomt.inference import load_model_for_eval, generate_action, generate_action_flat, extract_action_from_react

def get_next_action(
    model, 
    tokenizer, 
    method: str, 
    history_parts: list, 
    steps: int = 32, 
    gen_length: int = 256
) -> tuple:
    """Dispatches to either chat or flat inference format."""
    if method in ("standard_sft", "prefix_sft", "zero_shot"):
        # Chat format for SFT baselines
        prompt_text = "\n".join(history_parts)
        conversation = [{"role": "user", "content": prompt_text}]
        raw = generate_action(model, tokenizer, conversation, gen_length=gen_length, steps=steps)
    elif method == "aomt_mixed":
        # Flat format for AOMT
        raw = generate_action_flat(model, tokenizer, history_parts, gen_length=gen_length, steps=steps)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return raw, extract_action_from_react(raw)

def corrupt_observation(obs: str, rate: float) -> str:
    """Randomly mask tokens in the observation based on rate (rho)."""
    if rate <= 0: return obs
    words = obs.split()
    corrupted = [w if random.random() > rate else "[MASK]" for w in words]
    return " ".join(corrupted)

def evaluate_alfworld(
    model, tokenizer,
    config_path: str,
    method: str,
    split: str = "eval_out_of_distribution",
    num_episodes: int = None,
    steps: int = 32,
    gen_length: int = 256,
    obs_corruption_rate: float = 0.0,
    verbose: bool = True,
) -> dict:
    import alfworld.agents.environment as alfworld_env
    with open(config_path) as f:
        config = yaml.safe_load(f)

    env = alfworld_env.AlfredTWEnv(config, train_eval=split)
    env = env.init_env(batch_size=1)

    n_episodes = num_episodes or len(env.game_files)
    successes  = []

    for ep_idx in range(n_episodes):
        obs, infos = env.reset()
        obs = obs[0]
        if obs_corruption_rate > 0:
            obs = corrupt_observation(obs, obs_corruption_rate)

        history_parts = [obs]
        success = False

        for step_idx in range(50):
            raw_gen, action = get_next_action(model, tokenizer, method, history_parts, steps=steps, gen_length=gen_length)

            if verbose and ep_idx < 1:
                print(f"  [step={step_idx}] action: {action!r}")

            obs_next, scores, dones, infos = env.step([action])
            obs_next = obs_next[0]
            if obs_corruption_rate > 0:
                obs_next = corrupt_observation(obs_next, obs_corruption_rate)
            
            done = dones[0]
            score = scores[0]

            history_parts.append(raw_gen)
            history_parts.append(obs_next)

            if done:
                success = score > 0
                break

        successes.append(float(success))
        if verbose:
            print(f"Episode {ep_idx:3d}: {'SUCCESS' if success else 'FAIL'}")

    env.close()
    rate = sum(successes) / len(successes) * 100 if successes else 0.0
    return {"success_rate": rate, "n_episodes": len(successes)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True, choices=["zero_shot", "standard_sft", "prefix_sft", "aomt_mixed"])
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--benchmark", type=str, default="alfworld", choices=["alfworld", "scienceworld", "webshop"])
    parser.add_argument("--config", type=str, default="aomt/configs/alfworld_config.yaml")
    parser.add_argument("--split", type=str, default="eval_out_of_distribution")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--gen_length", type=int, default=256)
    parser.add_argument("--obs_corruption_rate", type=float, default=0.0)
    parser.add_argument("--output_json", type=str, default="results.json")
    args = parser.parse_args()

    tok_path = args.tokenizer if args.tokenizer else args.checkpoint_dir
    model, tokenizer, _ = load_model_for_eval(args.checkpoint_dir, tok_path)

    if args.benchmark == "alfworld":
        result = evaluate_alfworld(
            model, tokenizer, args.config, args.method,
            split=args.split, num_episodes=args.episodes,
            steps=args.steps, gen_length=args.gen_length,
            obs_corruption_rate=args.obs_corruption_rate
        )
    else:
        print(f"Benchmark {args.benchmark} environment not fully integrated in this smoke test.")
        result = {"success_rate": 0.0, "error": "Benchmark not implemented"}

    result["method"] = args.method
    result["benchmark"] = args.benchmark
    result["steps"] = args.steps
    
    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=4)
    print(f"\nResults saved to {args.output_json}")

if __name__ == "__main__": main()

