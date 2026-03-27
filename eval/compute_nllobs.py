import torch
import yaml
import json
import os
import argparse
from tqdm import tqdm
from aomt.inference import load_model_for_eval

def compute_nllobs(model, tokenizer, dataset_path, max_seq_length=2048):
    """
    Computes the Negative Log-Likelihood of observations given actions.
    This is AOMT's proxy metric for world model quality.
    """
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    sep_id = tokenizer.eos_token_id

    with open(dataset_path) as f:
        for line in f:
            ex = json.loads(line)
            unit_texts = ex["unit_texts"]
            unit_types = ex["unit_types"]

            all_ids = []
            spans = []
            for text, utype in zip(unit_texts, unit_types):
                ids = tokenizer.encode(text, add_special_tokens=False)
                start = len(all_ids)
                all_ids.extend(ids)
                end = len(all_ids)
                spans.append((start, end, utype))
                all_ids.append(sep_id)

            input_ids = torch.tensor([all_ids], dtype=torch.long).to(model.device)
            if input_ids.shape[1] > max_seq_length:
                input_ids = input_ids[:, :max_seq_length]

            # Mask observations for NLL calculation
            labels = torch.full_like(input_ids, -100)
            for start, end, utype in spans:
                if utype == "obs":
                    if start < max_seq_length:
                        actual_end = min(end, max_seq_length)
                        labels[0, start:actual_end] = input_ids[0, start:actual_end].clone()
                        input_ids[0, start:actual_end] = tokenizer.mask_token_id

            with torch.no_grad():
                logits = model(input_ids).logits
                # Shift for next-token prediction isn't needed here as we use MASK tokens,
                # but we align labels with logits directly.
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                    reduction="sum"
                )
                total_nll += loss.item()
                total_tokens += (labels != -100).sum().item()

    return total_nll / total_tokens if total_tokens > 0 else 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--benchmark", type=str, default="alfworld")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output_json", type=str, default="nllobs.json")
    args = parser.parse_args()

    model, tokenizer, _ = load_model_for_eval(args.checkpoint_dir, args.checkpoint_dir)
    
    # Map benchmark/split to cached jsonl path
    data_path = f"data/cache/aomt_{args.split}.jsonl"
    
    nll = compute_nllobs(model, tokenizer, data_path)
    
    result = {"nll_obs": nll, "checkpoint": args.checkpoint_dir, "benchmark": args.benchmark}
    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=4)
    print(f"NLLobs: {nll:.4f}")

if __name__ == "__main__": main()
