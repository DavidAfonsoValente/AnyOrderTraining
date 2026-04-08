import torch
import os
import json
import pandas as pd
from tqdm import tqdm
from ..evaluation.metrics import compute_observation_masked_nll
from ..data.dataset import AOMTDataset
from ..data.utils import load_robust_dataset
from ..model.llada_wrapper import load_model_and_tokenizer
from omegaconf import OmegaConf

def compute_all_nlls(checkpoint_dir, output_dir):
    """
    TABLE 2: nll_table.csv
    Computes Obs-masked NLL for AOMT-Mixed and baselines.
    """
    # Load dataset (test split)
    print("Loading test dataset for NLL computation...")
    ds_dict, _ = load_robust_dataset()
    test_raw = ds_dict["test"]
    
    methods = {
        "amx_p025": "AOMT-Mixed (p=0.25)",
        "std_sft": "Standard SFT"
    }
    
    results = []
    
    for m_key, m_name in methods.items():
        ckpt_path = os.path.join(checkpoint_dir, m_key)
        if not os.path.exists(ckpt_path):
            print(f"Skipping {m_name}, checkpoint not found.")
            continue
            
        print(f"Computing NLL for {m_name}...")
        model, tokenizer = load_model_and_tokenizer(model_id=ckpt_path)
        
        # Create dataset to get tokenized trajectories
        ds = AOMTDataset(
            raw_dataset=test_raw,
            tokenizer=tokenizer,
            method="aomt_mixed", # We use AMX format to get full trajectories for NLL
            model_id="llada2-mini"
        )
        
        test_trajs = ds.tokenized_trajectories
        
        try:
            nll = compute_observation_masked_nll(
                model=model,
                tokenizer=tokenizer,
                test_trajectories=test_trajs,
                method="aomt_mixed", # Force AMX logic for the metric
                device=model.device
            )
            results.append({"Method": m_name, "Obs-masked NLL": nll})
        except Exception as e:
            print(f"Error computing NLL for {m_key}: {e}")
            
    df = pd.DataFrame(results)
    output_path = os.path.join(output_dir, "nll_table.csv")
    df.to_csv(output_path, index=False)
    print(f"NLL table saved to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    compute_all_nlls(args.checkpoint_dir, args.output_dir)
