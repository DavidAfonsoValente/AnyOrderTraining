import os
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from veomni.models import build_model
from transform.trajectory_transform import TrajectoryTransform

def calculate_forward_nll(model_path, dataset_path, output_file):
    """
    Calculates forward (causal) Negative Log-Likelihood for observations and actions
    based on the project specification.
    """
    # 1. Load Model, Tokenizer, and Data Transform
    print(f"Loading model and tokenizer from: {model_path}")
    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = build_model(model_path, torch_dtype=dtype, trust_remote_code=True).cuda().eval()
    
    transformer = TrajectoryTransform(tokenizer)

    # 2. Load Dataset
    print(f"Loading dataset from: {dataset_path}")
    with open(dataset_path, 'r') as f:
        # Assuming each line is a JSON object with a 'trajectory' key
        trajectories = [json.loads(line)['trajectory'] for line in f]

    all_traj_obs_nlls = []
    all_traj_act_nlls = []

    # 3. Iterate through trajectories
    for trajectory in tqdm(trajectories, desc="Evaluating Forward NLL"):
        processed = transformer.process_trajectory(trajectory)
        
        input_ids = processed['input_ids'].cuda()
        unit_boundaries = processed['unit_boundaries']
        unit_types = processed['unit_types']

        # --- NLL_obs_forward ---
        # Formula: log p(O_t+1 | O_0, A_0, ..., O_t, A_t)
        traj_obs_nll_sum = 0.0
        traj_obs_count = 0
        for i in range(len(unit_boundaries) - 1):
            if unit_types[i] == 'action' and unit_types[i+1] == 'observation':
                context_end_idx = unit_boundaries[i][1]
                label_start_idx, label_end_idx = unit_boundaries[i+1]
                
                model_input_ids = input_ids[:label_end_idx].unsqueeze(0)
                labels = torch.full_like(model_input_ids, -100)
                labels[0, context_end_idx:label_end_idx] = input_ids[context_end_idx:label_end_idx]

                with torch.no_grad():
                    outputs = model(input_ids=model_input_ids, labels=labels)
                    # The loss is the mean NLL per token. Sum for the unit.
                    nll = outputs.loss.item() * (label_end_idx - context_end_idx)
                
                traj_obs_nll_sum += nll
                traj_obs_count += 1
        
        if traj_obs_count > 0:
            # Normalize by n_fwd(τ) as per spec
            all_traj_obs_nlls.append(traj_obs_nll_sum / traj_obs_count)

        # --- NLL_act_forward ---
        # Formula: log p(A_t | O_0, A_0, ..., O_t)
        traj_act_nll_sum = 0.0
        traj_act_count = 0
        for i in range(len(unit_boundaries)):
            if unit_types[i] == 'action':
                # Context is everything before this action unit
                context_end_idx = unit_boundaries[i-1][1] if i > 0 else 0
                label_start_idx, label_end_idx = unit_boundaries[i]
                
                model_input_ids = input_ids[:label_end_idx].unsqueeze(0)
                labels = torch.full_like(model_input_ids, -100)
                labels[0, context_end_idx:label_end_idx] = input_ids[context_end_idx:label_end_idx]

                with torch.no_grad():
                    outputs = model(input_ids=model_input_ids, labels=labels)
                    nll = outputs.loss.item() * (label_end_idx - context_end_idx)

                traj_act_nll_sum += nll
                traj_act_count += 1
        
        if traj_act_count > 0:
            # Normalize by n_act(τ) as per spec
            all_traj_act_nlls.append(traj_act_nll_sum / traj_act_count)
            
    # 4. Final Averaging and Saving
    # Formula: Average the per-trajectory normalized NLLs over the whole dataset |T|
    final_nll_obs_forward = np.mean(all_traj_obs_nlls) if all_traj_obs_nlls else 0
    final_nll_act_forward = np.mean(all_traj_act_nlls) if all_traj_act_nlls else 0

    print(f"
{'='*20} Evaluation Complete {'='*20}")
    print(f"NLL_obs_forward: {final_nll_obs_forward:.4f}")
    print(f"NLL_act_forward: {final_nll_act_forward:.4f}")
    print(f"{'='*59}")

    results = {
        "model_path": model_path,
        "dataset_path": dataset_path,
        "NLL_obs_forward": final_nll_obs_forward,
        "NLL_act_forward": final_nll_act_forward,
    }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate causal forward NLL for observations and actions.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model checkpoint.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the evaluation dataset (JSONL format).")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the output JSON results.")
    args = parser.parse_args()
    
    calculate_forward_nll(args.model_path, args.dataset_path, args.output_file)
