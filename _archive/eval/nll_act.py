# aomt/eval/nll_act.py
"""
eval/nll_act.py
Action-masked NLL evaluation for AOMT models.
Updated to use unified unit_parser and chat templates.
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import json
import argparse

from aomt.data.unit_parser import parse_conversation_to_trajectory, tokenize_trajectory
from transformers import AutoModelForCausalLM, AutoTokenizer

@torch.no_grad()
def compute_nll_act(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    examples: list,
    batch_size: int = 1, # Use 1 for simpler logic since trajectories vary in length
    device: str = "cuda"
) -> dict:
    """
    Computes the Action-Masked Negative Log-Likelihood (NLL-act).
    """
    model.eval()
    model.to(device)
    mask_token_id = tokenizer.mask_token_id or 156895

    results = {
        "all_nll": [],
        "nll_by_env": defaultdict(list),
        "nll_by_position": defaultdict(list),
    }

    for ex in tqdm(examples, desc="Evaluating NLL-act"):
        traj = parse_conversation_to_trajectory(ex)
        tokenized_traj = tokenize_trajectory(traj, tokenizer)
        
        if tokenized_traj is None:
            continue

        clean_ids = tokenized_traj.input_ids
        
        act_units = [u for u in tokenized_traj.unit_spans if u.unit_type == "act"]
        for act_unit in act_units:
            if act_unit.token_start >= act_unit.token_end:
                continue

            masked_ids = clean_ids.clone()
            masked_ids[act_unit.token_start:act_unit.token_end] = mask_token_id
            
            # Bidirectional attention
            attn_mask = torch.ones((1, len(masked_ids)), device=device)

            logits = model(
                input_ids=masked_ids.unsqueeze(0).to(device),
                attention_mask=attn_mask,
            ).logits[0]

            act_logits = logits[act_unit.token_start:act_unit.token_end]
            act_target = clean_ids[act_unit.token_start:act_unit.token_end].to(device)

            nll = F.cross_entropy(act_logits, act_target, reduction="mean").item()

            results["all_nll"].append(nll)
            results["nll_by_env"][tokenized_traj.env].append(nll)
            results["nll_by_position"][act_unit.unit_index // 2].append(nll)

    final_metrics = {
        "mean_nll_act": float(np.mean(results["all_nll"])) if results["all_nll"] else 0.0,
        "nll_by_env": {
            k: float(np.mean(v)) for k, v in results["nll_by_env"].items()
        },
        "nll_by_position": {
            k: float(np.mean(v)) for k, v in sorted(results["nll_by_position"].items())
        },
    }

    return final_metrics

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

    results = compute_nll_act(model, tokenizer, examples)
    print(f"Mean NLL-act: {results['mean_nll_act']:.4f}")

    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
