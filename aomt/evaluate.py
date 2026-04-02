import argparse
import os
import torch
import json
from omegaconf import OmegaConf
from aomt.model.llada_wrapper import load_model_and_tokenizer
from aomt.evaluation.eval_alfworld import evaluate_alfworld
from aomt.evaluation.metrics import compute_observation_masked_nll
from aomt.data.utils import load_robust_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--benchmark", type=str, choices=["alfworld", "scienceworld", "webshop", "all"])
    parser.add_argument("--inference_mode", type=str, default="mode_a", choices=["mode_a", "mode_b"])
    parser.add_argument("--planning_horizon", type=int, default=3)
    parser.add_argument("--rho", type=float, default=0.0)
    parser.add_argument("--output_dir", type=str, default="results/")
    args = parser.parse_args()

    # Load config from checkpoint if it exists, otherwise use base
    base_cfg = OmegaConf.load("aomt/config/base.yaml")
    # model_id override to checkpoint path
    base_cfg.model_id = args.checkpoint
    
    print(f"Loading model from {args.checkpoint}...")
    model, tokenizer = load_model_and_tokenizer(model_id=args.checkpoint)
    
    results = {}
    
    if args.benchmark in ["alfworld", "all"]:
        res = evaluate_alfworld(
            model, tokenizer, base_cfg, 
            inference_mode=args.inference_mode,
            planning_horizon=args.planning_horizon,
            rho=args.rho
        )
        results["alfworld"] = res
        
    # ... handle other benchmarks
    
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"results_{args.benchmark}_{args.inference_mode}_rho{args.rho}.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
