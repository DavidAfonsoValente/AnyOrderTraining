"""
eval/noise_robustness.py
Robustness evaluation under corrupted observations.
Implements Table 5 from the paper: randomly corrupts fraction rho of observation
tokens at test time and measures task success rate.
"""

import argparse
import json
import os
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

from aomt.eval.task_eval import (
    build_tokenizer,
    generate_action,
    SYSTEM_PROMPT,
)


def corrupt_observation(obs_text: str, rho: float, tokenizer, rng) -> str:
    """
    Corrupt fraction rho of observation tokens by replacing them with
    random vocabulary tokens, then decode back to text.
    
    Paper (Sec. 5.3): "randomly corrupting fraction rho in {0.1, 0.2, 0.3}
    of observation tokens at test time (ALFWorld seen split)."
    """
    if rho <= 0.0:
        return obs_text
    
    tokens = tokenizer.encode(obs_text, add_special_tokens=False)
    if len(tokens) == 0:
        return obs_text
    
    # Select which tokens to corrupt
    mask = rng.random(len(tokens)) < rho
    n_corrupt = int(mask.sum())
    
    if n_corrupt == 0:
        return obs_text
    
    # Replace corrupted positions with random vocab IDs
    vocab_size = len(tokenizer)
    random_ids = rng.integers(0, vocab_size, size=n_corrupt)
    
    corrupted = list(tokens)
    corrupt_idx = 0
    for i in range(len(corrupted)):
        if mask[i]:
            corrupted[i] = int(random_ids[corrupt_idx])
            corrupt_idx += 1
    
    return tokenizer.decode(corrupted, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# ALFWorld evaluation with corrupted observations
# ---------------------------------------------------------------------------

def evaluate_alfworld_noisy(
    model, tokenizer, rho: float, split="seen", n_episodes=50,
    max_steps=50, seed=42, **gen_kwargs
):
    """
    Runs ALFWorld evaluation with observation corruption at rate rho.
    Same as task_eval.evaluate_alfworld but corrupts each environment
    observation before adding it to the history.
    """
    import yaml
    from alfworld.agents.environment import get_environment

    rng = np.random.default_rng(seed)

    config_path = os.environ.get("ALFWORLD_CONFIG", "configs/alfworld_config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"ALFWorld config not found at {config_path}. Check $ALFWORLD_CONFIG."
        )

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    split_map = {
        "seen": "eval_in_distribution",
        "unseen": "eval_out_of_distribution",
        "test": "eval_out_of_distribution",
        "train": "train",
    }
    train_eval = split_map.get(split, "eval_in_distribution")

    env_type = config.get("env", {}).get("type", "AlfredTWEnv")
    EnvClass = get_environment(env_type)
    tw_env = EnvClass(config, train_eval=train_eval)
    gym_env = tw_env.init_env(batch_size=1)

    print(
        f"Running ALFWorld robustness evaluation "
        f"(ρ={rho}, split={split} -> {train_eval}, "
        f"{n_episodes} episodes, {tw_env.num_games} games)..."
    )

    successes = []
    for ep in tqdm(
        range(min(n_episodes, tw_env.num_games)), desc=f"ALFWorld ρ={rho}"
    ):
        obs, infos = gym_env.reset()
        if isinstance(obs, (list, tuple)):
            obs = obs[0]
        obs = str(obs)

        # System prompt is NOT corrupted (it's instructions, not observations)
        # The initial environment observation IS corrupted
        corrupted_obs = corrupt_observation(obs, rho, tokenizer, rng)
        obs_history = SYSTEM_PROMPT + "\nOK\n" + corrupted_obs
        done = False
        reward = 0.0

        for step in range(max_steps):
            if done:
                break
            action = generate_action(model, tokenizer, obs_history, **gen_kwargs)

            obs_list, reward_list, done_list, infos = gym_env.step([action])
            obs = obs_list[0] if isinstance(obs_list, (list, tuple)) else obs_list
            reward = (
                reward_list[0] if isinstance(reward_list, (list, tuple)) else reward_list
            )
            done = (
                done_list[0] if isinstance(done_list, (list, tuple)) else done_list
            )
            obs = str(obs)

            # Corrupt each new observation before appending to history
            corrupted_obs = corrupt_observation(obs, rho, tokenizer, rng)
            obs_history += f"\n{action}\n{corrupted_obs}"

        successes.append(1.0 if reward > 0 else 0.0)

    gym_env.close()
    return {
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "episodes": len(successes),
        "rho": rho,
        "split": split,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="AOMT Robustness Evaluation (Table 5)"
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the model checkpoint")
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="Path to the tokenizer")
    parser.add_argument("--benchmark", type=str, default="alfworld",
                        choices=["alfworld"],
                        help="Benchmark (currently ALFWorld only)")
    parser.add_argument("--split", type=str, default="seen",
                        choices=["seen", "unseen", "test", "train"])
    parser.add_argument("--rho", type=float, required=True,
                        help="Fraction of obs tokens to corrupt (e.g. 0.1, 0.2, 0.3)")
    parser.add_argument("--n_episodes", type=int, default=50)
    parser.add_argument("--gen_length", type=int, default=256)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.device:
        init_device = {"": args.device}
    elif torch.cuda.is_available():
        init_device = {"": "cuda:0"}
    else:
        init_device = "cpu"

    print(f"Loading model from {args.model_path}...")
    tokenizer = build_tokenizer(args.tokenizer)
    model = build_foundation_model(
        weights_path=args.model_path,
        config_path=args.model_path,
        torch_dtype="bfloat16",
        init_device=init_device,
    )
    model.eval()

    gen_params = {
        "gen_length": args.gen_length,
        "block_length": args.block_length,
        "steps": args.steps,
    }

    if args.benchmark == "alfworld":
        result = evaluate_alfworld_noisy(
            model, tokenizer,
            rho=args.rho,
            split=args.split,
            n_episodes=args.n_episodes,
            seed=args.seed,
            **gen_params,
        )
    else:
        raise ValueError(f"Unsupported benchmark: {args.benchmark}")

    print(f"\nRobustness Evaluation Complete ({args.benchmark}, ρ={args.rho})")
    print(json.dumps(result, indent=4))

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(result, f, indent=4)


if __name__ == "__main__":
    main()
