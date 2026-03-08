"""
eval/nll_obs.py
Observation-masked NLL evaluation for AOMT models.
Verified implementation based on Engineering Specification v3.
"""

import argparse
import json
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

@torch.no_grad()
def compute_nll_obs(model, tokenizer, aomt_examples: list) -> float:
    """
    For each obs unit O_t: mask it, keep all others as context, measure CE.
    Returns mean NLL over all obs units.
    """
    MASK = tokenizer.mask_token_id
    if MASK is None:
        MASK = 156895 # Fallback for LLaDA2.0-mini
    
    SEP = tokenizer.eos_token_id
    all_nll = []

    for ex in tqdm(aomt_examples, desc="Computing NLL-obs"):
        # Build token sequence
        all_ids, spans = [], []
        for i, (text, utype) in enumerate(zip(ex["unit_texts"], ex["unit_types"])):
            ids = tokenizer.encode(text, add_special_tokens=False)
            s = len(all_ids)
            all_ids.extend(ids)
            spans.append((s, len(all_ids), utype))
            if i < len(ex["unit_texts"]) - 1:
                all_ids.append(SEP)

        clean_ids = torch.tensor(all_ids, dtype=torch.long)

        for start, end, utype in spans:
            if utype != "obs":
                continue
            
            masked = clean_ids.clone()
            masked[start:end] = MASK
            
            # Bidirectional attention
            attn_mask = torch.ones((1, len(masked)), device=model.device)
            
            logits = model(
                input_ids=masked.unsqueeze(0).to(model.device),
                attention_mask=attn_mask
            ).logits[0]
            
            nll = F.cross_entropy(
                logits[start:end],
                clean_ids[start:end].to(model.device),
            ).item()
            all_nll.append(nll)

    return float(np.mean(all_nll))

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
