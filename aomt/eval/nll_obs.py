"""
eval/nll_obs.py
Observation-masked NLL evaluation for AOMT models.
Updated to use unified unit_parser and chat templates.
"""

import argparse
import json
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from aomt.data.unit_parser import parse_conversation_to_trajectory, tokenize_trajectory

@torch.no_grad()
def compute_nll_obs(model, tokenizer, aomt_examples: list) -> float:
    """
    For each obs unit O_t: mask it, keep all others as context, measure CE.
    Returns mean NLL over all obs units.
    """
    mask_token_id = tokenizer.mask_token_id or 156895
    all_nll = []

    for ex in tqdm(aomt_examples, desc="Computing NLL-obs"):
        # 1. Parse and Tokenize using unified logic (with chat template)
        traj = parse_conversation_to_trajectory(ex)
        tokenized_traj = tokenize_trajectory(traj, tokenizer)
        
        if tokenized_traj is None:
            continue

        clean_ids = tokenized_traj.input_ids
        
        # 2. Iterate over observation units (excluding the objective at index 0)
        for unit in tokenized_traj.unit_spans:
            if unit.unit_type != "obs" or unit.unit_index == 0:
                continue
            
            if unit.token_start >= unit.token_end:
                continue

            masked = clean_ids.clone()
            masked[unit.token_start:unit.token_end] = mask_token_id
            
            # Bidirectional attention
            attn_mask = torch.ones((1, len(masked)), device=model.device)
            
            logits = model(
                input_ids=masked.unsqueeze(0).to(model.device),
                attention_mask=attn_mask
            ).logits[0]
            
            # CE loss for the masked unit
            nll = F.cross_entropy(
                logits[unit.token_start:unit.token_end],
                clean_ids[unit.token_start:unit.token_end].to(model.device),
            ).item()
            all_nll.append(nll)

    return float(np.mean(all_nll)) if all_nll else 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    examples = []
    with open(args.data_path, "r") as f:
        for line in f:
            examples.append(json.loads(line))

    nll = compute_nll_obs(model, tokenizer, examples)
    print(f"Mean NLL-obs: {nll:.4f}")

    with open(args.output_file, "w") as f:
        json.dump({"nll_obs": nll}, f)

if __name__ == "__main__":
    main()
