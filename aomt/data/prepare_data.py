# aomt/data/prepare_data.py

import argparse
import json
import os
from datasets import load_dataset

def parse_trajectory(example: dict) -> list | None:
    """
    Parse one ETO example into a list of {"type": "obs"/"act", "text": str}.

    CRITICAL: ETO dataset uses {"from": "human"/"gpt", "value": "…"}
              NOT {"role": "user"/"assistant", "content": "…"}
    """
    units = []
    if "conversations" not in example:
        return None
    for turn in example["conversations"]:
        role  = turn["from"]    # "human" or "gpt" — NOT "role"
        text  = turn["value"]   # actual content    — NOT "content"
        utype = "obs" if role == "human" else "act"
        units.append({"type": utype, "text": text})

    # Validate strictly alternating obs/act structure
    for i, u in enumerate(units):
        expected = "obs" if i % 2 == 0 else "act"
        if u["type"] != expected:
            return None  # skip malformed
    return units

def make_standard_sft_examples(units: list, sep: str = "\n") -> list:
    """
    T examples per trajectory. Each example: predict A_t from full causal history.
    """
    examples = []
    history  = []

    for unit in units:
        if unit["type"] == "obs":
            history.append(unit["text"])
        else:  # "act"
            if not history:
                continue
            examples.append({
                "messages": [
                    {"role": "user",      "content": sep.join(history)},
                    {"role": "assistant", "content": unit["text"]},
                ]
            })
            history.append(unit["text"])

    return examples

def make_prefix_sft_s1_examples(units: list, sep: str = "\n") -> list:
    """
    T-1 examples per trajectory. Each example: predict O_{t+1} from (O_t, A_t).
    LOCAL context only — not full history.
    """
    examples = []
    for i in range(len(units) - 2):
        if (units[i]["type"] == "obs"
                and units[i+1]["type"] == "act"
                and units[i+2]["type"] == "obs"):
            examples.append({
                "messages": [
                    {"role": "user",
                     "content": sep.join([units[i]["text"], units[i+1]["text"]])},
                    {"role": "assistant", "content": units[i+2]["text"]},
                ]
            })
    return examples

def make_aomt_datapoint(units: list) -> dict:
    """One datapoint per trajectory. Masking happens at training time."""
    return {
        "unit_texts": [u["text"] for u in units],
        "unit_types": [u["type"] for u in units],
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./data/cache/")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading ETO dataset...")
    # Using the local dataset if available, otherwise from HF
    try:
        dataset = load_dataset("agent-eto/eto-sft-trajectory")
    except:
        print("Failed to load from HF, check if dataset is available locally or check your connection.")
        return

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

if __name__ == "__main__":
    main()
