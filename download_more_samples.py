from datasets import load_dataset
import json
import sys

# Add project root to path
sys.path.append("/home/davidvalente/AnyOrderTraining")
from aomt.data.utils import load_robust_dataset

split_ds, b_splits = load_robust_dataset()
samples = []
for domain, ds in b_splits.items():
    # Take 10 samples from each domain
    for i in range(min(10, len(ds))):
        samples.append(ds[i])

with open("/home/davidvalente/AnyOrderTraining/samples_large.json", "w") as f:
    json.dump(samples, f, indent=2)

print(f"Saved {len(samples)} samples to samples_large.json")
