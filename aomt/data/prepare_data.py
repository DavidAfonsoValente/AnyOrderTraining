# aomt/data/prepare_data.py

import os
import json
import argparse
from datasets import concatenate_datasets
from aomt.data.utils import load_robust_dataset
from aomt.data.parse_trajectories import parse_trajectory

def make_standard_sft_examples(units):
    """
    Standard SFT: Full causal history joined by \n.
    Messages: [user=O0\nA0...Ot, assistant=At]
    One example per action in the trajectory.
    """
    examples = []
    history = []
    for i in range(len(units)):
        if units[i]["unit_type"] == "obs":
            history.append(units[i]["text"])
        elif units[i]["unit_type"] == "act":
            # assistant turn
            examples.append({
                "messages": [
                    {"role": "user", "content": "\n".join(history)},
                    {"role": "assistant", "content": units[i]["text"]}
                ]
            })
            # Add action to history for next step
            history.append(units[i]["text"])
    return examples

def make_prefix_sft_s1_examples(units):
    """
    Prefix SFT Stage 1: Local pair only. 
    Messages: [user=Ot\nAt, assistant=Ot+1]
    """
    examples = []
    sep = "\n"
    for i in range(len(units) - 2):
        if units[i]["unit_type"] == "obs" and units[i+1]["unit_type"] == "act" and units[i+2]["unit_type"] == "obs":
            examples.append({
                "messages": [
                    {"role": "user", 
                     "content": sep.join([units[i]["text"], units[i+1]["text"]])},
                    {"role": "assistant", "content": units[i+2]["text"]},
                ]
            })
    return examples

def make_aomt_datapoint(units):
    """
    AOMT: Flat trajectory.
    """
    return {
        "unit_texts": [u["text"] for u in units],
        "unit_types": [u["unit_type"] for u in units]
    }

def process_and_save():
    # Merge
    all_ds, alfworld_ds, sciworld_ds, webshop_ds = load_robust_dataset()
    combined = concatenate_datasets(all_ds)
    
    # Split into train/test (90/10)
    split_ds = combined.train_test_split(test_size=0.1, seed=42)
    
    # Also return individual benchmark datasets for test-set NLL_obs
    benchmarks = {
        "alfworld": alfworld_ds.train_test_split(test_size=0.1, seed=42)["test"],
        "scienceworld": sciworld_ds.train_test_split(test_size=0.1, seed=42)["test"],
        "webshop": webshop_ds.train_test_split(test_size=0.1, seed=42)["test"]
    }
    
    return split_ds, benchmarks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./data/cache/")
    parser.add_argument("--tokenizer", type=str, help="Path to tokenizer (optional for this script)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading ETO dataset via robust loader...")
    try:
        dataset, benchmark_tests = process_and_save()
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    # 1. Main Splits
    for split in ["train", "test"]:
        print(f"Processing {split} split...")
        sft_data = []
        prefix_s1_data = []
        aomt_data = []

        for ex in dataset[split]:
            units = parse_trajectory(ex)
            if not units:
                continue
            
            sft_data.extend(make_standard_sft_examples(units))
            prefix_s1_data.extend(make_prefix_sft_s1_examples(units))
            aomt_data.append(make_aomt_datapoint(units))

        # Write files
        for name, data in [
            ("sft_standard", sft_data),
            ("prefix_sft_s1", prefix_s1_data),
            ("aomt", aomt_data)
        ]:
            out_path = os.path.join(args.output_dir, f"{name}_{split}.jsonl")
            with open(out_path, "w") as f:
                for entry in data:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            print(f"  Wrote {len(data)} entries to {out_path}")

    # 2. Benchmark-specific AOMT test sets (Required for Table 4)
    for b_name, b_ds in benchmark_tests.items():
        out_path = os.path.join(args.output_dir, f"{b_name}_aomt_test.jsonl")
        count = 0
        with open(out_path, "w") as f:
            for ex in b_ds:
                units = parse_trajectory(ex)
                if not units: continue
                f.write(json.dumps(make_aomt_datapoint(units), ensure_ascii=False) + "\n")
                count += 1
        print(f"  Wrote {count} entries to {out_path}")

if __name__ == "__main__":
    main()
