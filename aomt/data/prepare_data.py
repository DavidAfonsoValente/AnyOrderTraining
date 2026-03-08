"""
data/prepare_data.py
Converts raw expert trajectories (agent-eto/eto-sft-trajectory) into JSONL files 
per training mode: Standard SFT, Prefix SFT Stage 1, and AOMT.
"""

import argparse
import json
import os
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer
from data.utils import load_robust_dataset

def parse_units(conversations):
    """Strictly alternating obs/act units. First turn always obs."""
    units = []
    for turn in conversations:
        utype = "obs" if turn["from"] == "human" else "act"
        units.append({"type": utype, "text": turn["value"]})
    assert all(u["type"] == ("obs" if i % 2 == 0 else "act")
               for i, u in enumerate(units)), "Non-alternating trajectory"
    return units

def make_standard_sft(units, sep="\n"):
    """
    One datapoint per action. Prompt = full causal history up to O_t.
    Response = A_t.
    """
    datapoints = []
    prompt_parts = []
    for unit in units:
        if unit["type"] == "obs":
            prompt_parts.append(unit["text"])
        else:
            datapoints.append({
                "messages": [
                    {"role": "user",      "content": sep.join(prompt_parts)},
                    {"role": "assistant", "content": unit["text"]},
                ]
            })
            prompt_parts.append(unit["text"])
    return datapoints

def make_prefix_sft_s1(units, sep="\n"):
    """
    One datapoint per (O_t, A_t, O_{t+1}) triple.
    Prompt = (O_t, A_t) local context ONLY.
    Response = O_{t+1}.
    """
    datapoints = []
    for i in range(len(units) - 2):
        if (units[i]["type"] == "obs" and
            units[i+1]["type"] == "act" and
            units[i+2]["type"] == "obs"):
            datapoints.append({
                "messages": [
                    {"role": "user",
                     "content": units[i]["text"] + sep + units[i+1]["text"]},
                    {"role": "assistant", "content": units[i+2]["text"]},
                ]
            })
    return datapoints

def make_aomt_datapoint(units):
    """One datapoint per full trajectory."""
    return {
        "unit_texts": [u["text"] for u in units],
        "unit_types": [u["type"] for u in units],
    }

def main():
    parser = argparse.ArgumentParser(description="Prepare data for AOMT training")
    parser.add_argument("--raw_dir", type=str, default="./data/raw/", help="Path to raw dataset")
    parser.add_argument("--output_dir", type=str, default="./data/cache/", help="Path to save processed JSONL files")
    parser.add_argument("--tokenizer", type=str, required=True, help="Path to tokenizer")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dataset = load_robust_dataset()

    for split in ["train", "test"]:
        print(f"Processing {split} split...")
        sft_data = []
        prefix_s1_data = []
        aomt_data = []

        for ex in dataset[split]:
            units = parse_units(ex["conversations"])
            
            sft_data.extend(make_standard_sft(units))
            prefix_s1_data.extend(make_prefix_sft_s1(units))
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

if __name__ == "__main__":
    main()
