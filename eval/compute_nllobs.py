import torch
import torch.nn.functional as F
import yaml
import json
import os
import argparse
import numpy as np
from tqdm import tqdm
from aomt.inference import load_model_for_eval

def compute_nllobs(model, tokenizer, dataset_path, max_seq_length=2048, device="cuda"):
    """
    Computes the Negative Log-Likelihood of observations given actions.
    Paper formula: NLL_obs = -E_{τ} E_t [ log p_θ(O_t | τ \ O_t) ]
    Implementation: Mask ONE observation at a time, keep all other units clean.
    Uses FLAT trajectory format matching AOMT training.
    """
    model.eval()
    all_nlls = []
    sep_id = tokenizer.eos_token_id
    mask_id = tokenizer.mask_token_id

    with open(dataset_path) as f:
        lines = [line for line in f if line.strip()]

    for line in tqdm(lines, desc="Computing NLLobs"):
        ex = json.loads(line)
        unit_texts = ex["unit_texts"]
        unit_types = ex["unit_types"]

        # Build full clean trajectory flat
        all_ids = []
        spans = []
        for text, utype in zip(unit_texts, unit_types):
            ids = tokenizer.encode(text, add_special_tokens=False)
            if not ids: continue
            start = len(all_ids)
            all_ids.extend(ids)
            end = len(all_ids)
            spans.append((start, end, utype))
            all_ids.append(sep_id)

        clean_ids = torch.tensor(all_ids, dtype=torch.long, device=device)
        if clean_ids.shape[0] > max_seq_length:
            clean_ids = clean_ids[:max_seq_length]
            # Update spans to only include those within truncated length
            spans = [(s, e, t) for s, e, t in spans if e <= max_seq_length]

        # For each observation unit, mask it and compute NLL
        for start, end, utype in spans:
            if utype != "obs":
                continue
            
            # Mask current observation only
            input_ids = clean_ids.clone()
            input_ids[start:end] = mask_id
            
            # Label: only current observation
            labels = torch.full_like(clean_ids, -100)
            labels[start:end] = clean_ids[start:end]
            
            with torch.no_grad():
                logits = model(input_ids.unsqueeze(0)).logits
                # CE at masked positions
                loss = F.cross_entropy(
                    logits[0, start:end],
                    clean_ids[start:end],
                    reduction="mean"
                )
                all_nlls.append(loss.item())

    return float(np.mean(all_nlls)) if all_nlls else 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--benchmark", type=str, default="alfworld")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    model, tokenizer, _ = load_model_for_eval(args.checkpoint_dir, args.checkpoint_dir)
    
    # Map benchmark to specific aomt test file
    # Ensure data/cache/{benchmark}_aomt_test.jsonl exists
    data_path = f"data/cache/{args.benchmark}_aomt_test.jsonl"
    if not os.path.exists(data_path):
        # Fallback to generic aomt_test.jsonl if benchmark-specific one is missing
        data_path = f"data/cache/aomt_test.jsonl"

    nll = compute_nllobs(model, tokenizer, data_path, device=args.device)
    
    result = {
        "nll_obs": nll, 
        "checkpoint": args.checkpoint_dir, 
        "benchmark": args.benchmark,
        "split": args.split
    }
    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=4)
    print(f"\nNLLobs for {args.benchmark} ({args.split}): {nll:.4f}")

if __name__ == "__main__": main()
