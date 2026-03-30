# aomt/data/prepare_data.py

import os
import json
import argparse
from aomt.data.utils import load_robust_dataset
from aomt.data.unit_parser import parse_conversation_to_trajectory

def parse_trajectory(example):
    """Wrap unit_parser to return a simple list of unit dicts for this script."""
    try:
        traj = parse_conversation_to_trajectory(example)
        return [{"unit_type": u.unit_type, "text": u.text} for u in traj.units]
    except Exception as e:
        print(f"Skipping example due to error: {e}")
        return []

def stream_standard_sft_examples(units, file_obj):
    """
    Standard SFT: Full causal history joined by \n.
    Writes one example per action directly to file_obj.
    """
    history = []
    count = 0
    for i in range(len(units)):
        if units[i]["unit_type"] == "obs":
            history.append(units[i]["text"])
        elif units[i]["unit_type"] == "act":
            entry = {
                "messages": [
                    {"role": "user", "content": "\n".join(history)},
                    {"role": "assistant", "content": units[i]["text"]}
                ]
            }
            file_obj.write(json.dumps(entry, ensure_ascii=False) + "\n")
            history.append(units[i]["text"])
            count += 1
    return count

def stream_prefix_sft_s1_examples(units, file_obj):
    """
    Prefix SFT Stage 1: Local pair only. 
    Writes entries directly to file_obj.
    """
    count = 0
    sep = "\n"
    for i in range(len(units) - 2):
        if units[i]["unit_type"] == "obs" and units[i+1]["unit_type"] == "act" and units[i+2]["unit_type"] == "obs":
            entry = {
                "messages": [
                    {"role": "user", 
                     "content": sep.join([units[i]["text"], units[i+1]["text"]])},
                    {"role": "assistant", "content": units[i+2]["text"]},
                ]
            }
            file_obj.write(json.dumps(entry, ensure_ascii=False) + "\n")
            count += 1
    return count

def stream_aomt_datapoint(units, file_obj):
    """
    AOMT: Flat trajectory.
    Writes entry directly to file_obj.
    """
    entry = {
        "unit_texts": [u["text"] for u in units],
        "unit_types": [u["unit_type"] for u in units]
    }
    file_obj.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./data/cache/")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading ETO dataset via robust loader...")
    try:
        dataset, benchmark_tests = load_robust_dataset()
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    # 1. Main Splits (Streaming to avoid MemoryError)
    for split in ["train", "test"]:
        print(f"Processing {split} split...")
        
        paths = {
            "sft_standard": os.path.join(args.output_dir, f"sft_standard_{split}.jsonl"),
            "prefix_sft_s1": os.path.join(args.output_dir, f"prefix_sft_s1_{split}.jsonl"),
            "aomt": os.path.join(args.output_dir, f"aomt_{split}.jsonl")
        }
        
        counts = {"sft_standard": 0, "prefix_sft_s1": 0, "aomt": 0}
        
        # Open all files for this split
        with open(paths["sft_standard"], "w") as f_sft, \
             open(paths["prefix_sft_s1"], "w") as f_pre, \
             open(paths["aomt"], "w") as f_aomt:
            
            for ex in dataset[split]:
                units = parse_trajectory(ex)
                if not units:
                    continue
                
                counts["sft_standard"] += stream_standard_sft_examples(units, f_sft)
                counts["prefix_sft_s1"] += stream_prefix_sft_s1_examples(units, f_pre)
                counts["aomt"] += stream_aomt_datapoint(units, f_aomt)

        for name, count in counts.items():
            print(f"  Wrote {count} entries to {paths[name]}")

    # 2. Benchmark-specific AOMT test sets (Required for Table 4)
    for b_name, b_ds in benchmark_tests.items():
        out_path = os.path.join(args.output_dir, f"{b_name}_aomt_test.jsonl")
        count = 0
        with open(out_path, "w") as f:
            for ex in b_ds:
                units = parse_trajectory(ex)
                if not units: continue
                count += stream_aomt_datapoint(units, f)
        print(f"  Wrote {count} entries to {out_path}")

if __name__ == "__main__":
    main()
