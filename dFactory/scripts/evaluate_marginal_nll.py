import os
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from veomni.models import build_model
from transform.trajectory_transform import TrajectoryTransform

def calculate_marginal_nll(model_path, dataset_path, output_file):
    """
    Calculates marginal (leave-one-unit-out) Negative Log-Likelihood.
    """
    # 1. Load Model and Dependencies
    print(f"Loading model and tokenizer from: {model_path}")
    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = build_model(model_path, torch_dtype=dtype, trust_remote_code=True).cuda().eval()
    transformer = TrajectoryTransform(tokenizer)

    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        # Fallback for tokenizers that might not have a specific mask_token
        mask_token_id = tokenizer.eos_token_id
        print(f"Warning: tokenizer.mask_token_id not found. Using eos_token_id ({mask_token_id}) as a fallback.")


    # 2. Load Dataset
    print(f"Loading dataset from: {dataset_path}")
    with open(dataset_path, 'r') as f:
        trajectories = [json.loads(line)['trajectory'] for line in f]

    all_obs_nlls = []
    all_act_nlls = []

    # 3. Iterate through trajectories
    for trajectory in tqdm(trajectories, desc="Evaluating Marginal NLL"):
        processed = transformer.process_trajectory(trajectory)
        
        original_input_ids = processed['input_ids'].cuda()
        unit_boundaries = processed['unit_boundaries']
        unit_types = processed['unit_types']

        # Iterate through each unit to calculate its NLL given the others
        for i, (start_idx, end_idx) in enumerate(unit_boundaries):
            unit_len = end_idx - start_idx
            if unit_len == 0:
                continue

            # Create the masked input
            masked_input_ids = original_input_ids.clone()
            masked_input_ids[start_idx:end_idx] = mask_token_id
            
            # Create labels: -100 everywhere except the masked unit
            labels = torch.full_like(original_input_ids, -100)
            labels[start_idx:end_idx] = original_input_ids[start_idx:end_idx]

            with torch.no_grad():
                outputs = model(input_ids=masked_input_ids.unsqueeze(0), labels=labels.unsqueeze(0))
                # The loss is the mean NLL per token. We want the sum for the unit.
                nll = outputs.loss.item() * unit_len

            if unit_types[i] == 'observation':
                all_obs_nlls.append(nll)
            elif unit_types[i] == 'action':
                all_act_nlls.append(nll)

    # 4. Final Averaging and Saving
    # As per spec, these are averages of the per-unit NLLs
    nll_obs = np.mean(all_obs_nlls) if all_obs_nlls else 0
    nll_act = np.mean(all_act_nlls) if all_act_nlls else 0
    nll_total = np.mean(all_obs_nlls + all_act_nlls) if (all_obs_nlls + all_act_nlls) else 0

    print(f"
{'='*20} Evaluation Complete {'='*20}")
    print(f"NLL_obs (LOO): {nll_obs:.4f}")
    print(f"NLL_act (LOO): {nll_act:.4f}")
    print(f"NLL_total (LOO): {nll_total:.4f}")
    print(f"{'='*59}")

    results = {
        "model_path": model_path,
        "dataset_path": dataset_path,
        "NLL_obs_total": nll_obs,
        "NLL_act_total": nll_act,
        "NLL_total": nll_total,
    }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate marginal (leave-one-out) NLL.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model checkpoint.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the evaluation dataset (JSONL format).")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the output JSON results.")
    args = parser.parse_args()
    
    calculate_marginal_nll(args.model_path, args.dataset_path, args.output_file)
