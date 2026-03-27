import json
import os

os.makedirs("data/cache", exist_ok=True)

# 1. Standard SFT Mock Data
sft_example = {
    "messages": [
        {"role": "user", "content": "You are in a kitchen. A potato is on the floor."},
        {"role": "assistant", "content": "Thought: I need to pick up the potato.\nAction: pick up potato"}
    ]
}

for split in ["train", "test"]:
    with open(f"data/cache/sft_standard_{split}.jsonl", "w") as f:
        for _ in range(10):
            f.write(json.dumps(sft_example) + "\n")

# 2. Prefix SFT Stage 1 Mock Data
prefix_example = {
    "messages": [
        {"role": "user", "content": "You are in a kitchen.\npick up potato"},
        {"role": "assistant", "content": "You picked up the potato."}
    ]
}

with open("data/cache/prefix_sft_s1_train.jsonl", "w") as f:
    for _ in range(10):
        f.write(json.dumps(prefix_example) + "\n")

# 3. AOMT Mock Data
aomt_example = {
    "unit_texts": [
        "You are in a kitchen. A potato is on the floor.",
        "pick up potato",
        "You picked up the potato.",
        "go to counter"
    ],
    "unit_types": ["obs", "act", "obs", "act"]
}

for split in ["train", "test"]:
    with open(f"data/cache/aomt_{split}.jsonl", "w") as f:
        for _ in range(10):
            f.write(json.dumps(aomt_example) + "\n")

print("Mock data generated in data/cache/")
