import argparse
import os
import torch
import json
from omegaconf import OmegaConf
from aomt.model.llada_wrapper import load_model_and_tokenizer
from aomt.evaluation.eval_alfworld import evaluate_alfworld
from aomt.evaluation.eval_scienceworld import evaluate_scienceworld
from aomt.evaluation.eval_webshop import evaluate_webshop
from aomt.evaluation.metrics import compute_observation_masked_nll
from aomt.data.utils import load_robust_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--benchmark", type=str, choices=["alfworld", "scienceworld", "webshop", "all"])
    parser.add_argument("--rho", type=float, default=0.0)
    parser.add_argument("--diffusion_steps", type=int, help="Override number of diffusion steps")
    parser.add_argument("--output_dir", type=str, default="results/")
    parser.add_argument("--n_episodes", type=int, default=50)
    args = parser.parse_args()

    # Load config from checkpoint if it exists, otherwise use base
    base_cfg = OmegaConf.load("aomt/config/base.yaml")
    # model_id override to checkpoint path
    base_cfg.model_id = args.checkpoint
    if args.diffusion_steps:
        base_cfg.diffusion_steps = args.diffusion_steps
    
    print(f"Loading model from {args.checkpoint}...")
    model, tokenizer = load_model_and_tokenizer(model_id=args.checkpoint)
    
    results = {}
    
    if args.benchmark in ["alfworld", "all"]:
        res = evaluate_alfworld(
            model, tokenizer, base_cfg, 
            rho=args.rho,
            n_episodes=args.n_episodes
        )
        results["alfworld"] = res
        
    if args.benchmark in ["scienceworld", "all"]:
        res = evaluate_scienceworld(
            model, tokenizer, base_cfg,
            n_episodes_per_task=1 # default for fast eval
        )
        results["scienceworld"] = res

    if args.benchmark in ["webshop", "all"]:
        res = evaluate_webshop(
            model, tokenizer, base_cfg,
            n_sessions=args.n_episodes
        )
        results["webshop"] = res
    
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"results_{args.benchmark}_rho{args.rho}.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
