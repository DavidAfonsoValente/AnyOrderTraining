from datasets import load_dataset
import json

splits = ['webshop', 'scienceworld', 'alfworld']
samples = []
for split in splits:
    ds = load_dataset("agent-eto/eto-sft-trajectory", split=split, streaming=True)
    for ex in ds:
        samples.append(ex)
        break

with open("samples.json", "w") as f:
    json.dump(samples, f, indent=2)

print(f"Downloaded {len(samples)} samples to samples.json")
