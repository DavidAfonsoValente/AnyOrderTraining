import sys
import os
import json
from datasets import load_dataset, Value

# Add project root to path
sys.path.append("/home/davidvalente/AnyOrderTraining")

from aomt.data.utils import load_robust_dataset

try:
    # Use streaming=False to get a proper dataset for selection
    # But load_robust_dataset doesn't take streaming.
    split_ds, b_splits = load_robust_dataset()
    
    samples = []
    # Take 1 sample from each domain test split if available
    for domain, ds in b_splits.items():
        if len(ds) > 0:
            samples.append(ds[0])
            print(f"Added sample from {domain}")

    with open("/home/davidvalente/AnyOrderTraining/samples.json", "w") as f:
        json.dump(samples, f, indent=2)

    print(f"Saved {len(samples)} samples to samples.json")

except Exception as e:
    print(f"Error: {e}")
    # Fallback to a very simple load
    print("Trying fallback loading...")
    base_url = "https://huggingface.co/datasets/agent-eto/eto-sft-trajectory/resolve/main/data"
    ds = load_dataset("json", data_files=f"{base_url}/alfworld_sft.json", split="train")
    sample = ds[0]
    with open("/home/davidvalente/AnyOrderTraining/samples.json", "w") as f:
        json.dump([sample], f, indent=2)
    print("Saved 1 fallback sample to samples.json")
