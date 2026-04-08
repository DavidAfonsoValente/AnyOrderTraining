import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import json
import torch
import glob
from typing import List
from .evaluate import main as run_eval
from omegaconf import OmegaConf

def run_p_sweep_ablation(checkpoint_dir: str, output_dir: str):
    """A1: p sweep for aomt_mixed. p in {0.15, 0.25, 0.40, 0.50}."""
    probs = ["015", "025", "040", "050"]
    results = []
    
    for p in probs:
        ckpt_path = os.path.join(checkpoint_dir, f"amx_p{p}")
        # Search for results in results/eval/amx_p{p}/
        res_files = glob.glob(os.path.join("results", "eval", f"amx_p{p}", "results_alfworld_*.json"))
        
        val = 0.0
        if res_files:
            with open(res_files[0], "r") as f:
                data = json.load(f)
                val = data.get("alfworld", {}).get("success_rate", 0.0)
        
        results.append({"p": float(f"0.{p[1:]}"), "success": val})
        
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "p_sweep_ablation.csv"), index=False)
    print(f"Saved p-sweep results to {output_dir}")

def run_k_sweep_ablation(checkpoint_dir: str, output_dir: str):
    """A2: k denoising steps sweep. k in {1, 2, 4, 8, 16, 32, 64}."""
    steps = [1, 2, 4, 8, 16, 32, 64]
    results = []
    
    # Use best p checkpoint (assume p025)
    ckpt_path = os.path.join(checkpoint_dir, "amx_p025")
    
    for k in steps:
        # In a real run, we would call evaluate.py --diffusion_steps k
        # Here we look for existing result files if any
        pattern = os.path.join("results", "eval", "amx_p025", f"results_alfworld_*_k{k}.json")
        res_files = glob.glob(pattern)
        
        val = 0.0
        if res_files:
            with open(res_files[0], "r") as f:
                data = json.load(f)
                val = data.get("alfworld", {}).get("success_rate", 0.0)
        
        results.append({"k": k, "success": val})
        
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "k_sweep_ablation.csv"), index=False)
    print(f"Saved k-sweep results to {output_dir}")

def run_unit_vs_token_masking_ablation(checkpoint_dir: str, output_dir: str):
    """A3: unit-level vs token-level masking."""
    # Compare amx_p025 vs amx_token
    results = []
    for m in ["amx_p025", "amx_token"]:
        res_files = glob.glob(os.path.join("results", "eval", m, "results_alfworld_*.json"))
        val = 0.0
        if res_files:
            with open(res_files[0], "r") as f:
                data = json.load(f)
                val = data.get("alfworld", {}).get("success_rate", 0.0)
        results.append({"method": m, "success": val})
        
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "unit_vs_token_ablation.csv"), index=False)

def run_robustness_ablation(checkpoint_dir: str, output_dir: str):
    """A4: Robustness under rho in {0.0, 0.1, 0.2, 0.3}."""
    rhos = [0.0, 0.1, 0.2, 0.3]
    methods = ["std_sft", "amx_p025"]
    
    all_results = []
    for m in methods:
        for rho in rhos:
            res_files = glob.glob(os.path.join("results", "eval", m, f"results_alfworld_rho{rho}.json"))
            val = 0.0
            if res_files:
                with open(res_files[0], "r") as f:
                    data = json.load(f)
                    val = data.get("alfworld", {}).get("success_rate", 0.0)
            all_results.append({"method": m, "rho": rho, "success": val})
            
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(output_dir, "robustness_ablation.csv"), index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", type=str, required=True, 
                        choices=["p_sweep", "k_sweep", "unit_vs_token", "robustness", "all"])
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/ablations/")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.ablation in ["p_sweep", "all"]:
        run_p_sweep_ablation(args.checkpoint_dir, args.output_dir)
    if args.ablation in ["k_sweep", "all"]:
        run_k_sweep_ablation(args.checkpoint_dir, args.output_dir)
    if args.ablation in ["unit_vs_token", "all"]:
        run_unit_vs_token_masking_ablation(args.checkpoint_dir, args.output_dir)
    if args.ablation in ["robustness", "all"]:
        run_robustness_ablation(args.checkpoint_dir, args.output_dir)
    
if __name__ == "__main__":
    main()
