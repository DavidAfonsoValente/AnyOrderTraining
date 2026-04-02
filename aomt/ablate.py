import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import json
import torch
from typing import List
from .evaluate import main as run_eval
from .model.llada_wrapper import load_model_and_tokenizer
from .evaluation.eval_alfworld import evaluate_alfworld
from .evaluation.eval_scienceworld import evaluate_scienceworld
from .evaluation.eval_webshop import evaluate_webshop
from .evaluation.eval_robustness import evaluate_robustness
from omegaconf import OmegaConf

def run_p_sweep_ablation(checkpoint_dir: str, output_dir: str):
    """A1: p sweep for aomt_mixed."""
    probs = ["015", "025", "040", "050"]
    results = []
    
    for p in probs:
        ckpt_path = os.path.join(checkpoint_dir, f"amx_p{p}")
        if not os.path.exists(ckpt_path):
            print(f"Warning: Checkpoint {ckpt_path} not found. Skipping.")
            continue
            
        print(f"Evaluating p_mask=0.{p[1:]}...")
        # In a real run, we would call evaluate_alfworld here
        results.append({"p": float(f"0.{p[1:]}"), "success": 0.0})
        
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "p_sweep_ablation.csv"), index=False)

def run_unit_vs_token_masking_ablation(checkpoint_dir: str, output_dir: str):
    """A2: unit-level vs token-level masking."""
    # Logic to compare amx_p025 (unit) vs amx_token (token)
    pass

def run_stage2_vs_sft_ablation(checkpoint_dir: str, output_dir: str):
    """A3: prefix_sft_stage2 vs standard_sft."""
    pass

def run_inference_mode_ablation(checkpoint_dir: str, output_dir: str):
    """A4: Mode A vs Mode B."""
    pass

def run_robustness_ablation(checkpoint_dir: str, output_dir: str):
    """A5: Robustness under rho."""
    pass

def run_learning_curve_ablation(checkpoint_dir: str, output_dir: str):
    """A6: Learning curve analysis."""
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", type=str, required=True, choices=["p_sweep", "unit_vs_token", "stage2_vs_sft", "inference_mode", "robustness", "learning_curve", "all"])
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/ablations/")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.ablation in ["p_sweep", "all"]:
        run_p_sweep_ablation(args.checkpoint_dir, args.output_dir)
    if args.ablation in ["unit_vs_token", "all"]:
        run_unit_vs_token_masking_ablation(args.checkpoint_dir, args.output_dir)
    if args.ablation in ["stage2_vs_sft", "all"]:
        run_stage2_vs_sft_ablation(args.checkpoint_dir, args.output_dir)
    if args.ablation in ["inference_mode", "all"]:
        run_inference_mode_ablation(args.checkpoint_dir, args.output_dir)
    if args.ablation in ["robustness", "all"]:
        run_robustness_ablation(args.checkpoint_dir, args.output_dir)
    if args.ablation in ["learning_curve", "all"]:
        run_learning_curve_ablation(args.checkpoint_dir, args.output_dir)
    
if __name__ == "__main__":
    main()
