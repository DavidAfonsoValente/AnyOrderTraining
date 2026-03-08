"""
data/measure_lengths.py
REQUIRED pre-training: validate that gen_length=256 covers all actions
and max_seq_length=2048 covers full trajectories.
"""

import argparse
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

def parse_units(conversations):
    """Strictly alternating obs/act units. First turn always obs."""
    units = []
    for turn in conversations:
        utype = "obs" if turn["from"] == "human" else "act"
        units.append({"type": utype, "text": turn["value"]})
    assert all(u["type"] == ("obs" if i % 2 == 0 else "act")
               for i, u in enumerate(units)), "Non-alternating trajectory"
    return units

def main():
    parser = argparse.ArgumentParser(description="Measure token lengths in dataset")
    parser.add_argument("--tokenizer", type=str, required=True, help="Path to tokenizer")
    parser.add_argument("--gen_length", type=int, default=256, help="Target generation length")
    parser.add_argument("--max_seq_len", type=int, default=2048, help="Target max sequence length")
    parser.add_argument("--raw_dir", type=str, default="./data/raw/", help="Path to raw dataset")
    args = parser.parse_args()

    print(f"Loading tokenizer from {args.tokenizer}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    
    print("Loading dataset agent-eto/eto-sft-trajectory...")
    dataset = load_dataset("agent-eto/eto-sft-trajectory")

    action_lengths, obs_lengths, traj_lengths = [], [], []

    for split in ["train", "test"]:
        print(f"Analyzing {split} split...")
        for ex in dataset[split]:
            units = parse_units(ex["conversations"])
            traj_len = 0
            for u in units:
                toks = len(tokenizer.encode(u["text"], add_special_tokens=False))
                traj_len += toks
                (action_lengths if u["type"] == "act" else obs_lengths).append(toks)
            traj_lengths.append(traj_len)

    failed = False
    for name, arr in [("Actions", action_lengths),
                      ("Observations", obs_lengths),
                      ("Full trajectories", traj_lengths)]:
        arr = np.array(arr)
        print(f"\n{name:20s}: mean={arr.mean():6.0f}  "
              f"p95={np.percentile(arr, 95):6.0f}  "
              f"p99={np.percentile(arr, 99):6.0f}  max={arr.max():6.0f}")

        if name == "Actions" and arr.max() >= args.gen_length:
            print(f"  CRITICAL: Max action length {arr.max()} exceeds gen_length={args.gen_length}")
            failed = True
        
        if name == "Full trajectories" and np.percentile(arr, 99) > args.max_seq_len:
            print(f"  WARNING: p99 trajectory length exceeds max_seq_len={args.max_seq_len}")

    if failed:
        print("\nLength validation FAILED. Increase gen_length before proceeding.")
        exit(1)
    else:
        print("\nLength validation PASSED.")

if __name__ == "__main__":
    main()
