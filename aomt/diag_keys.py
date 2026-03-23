#!/usr/bin/env python3
"""Diagnose weight key mismatch between base model and checkpoint."""
import os, sys, json, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Compare key names between base model index and checkpoint index
BASE = "./weights/llada2-mini-merged/model.safetensors.index.json"
CKPT = "./checkpoints/sft_standard/epoch_2/model.safetensors.index.json"

with open(BASE) as f:
    base_keys = set(json.load(f)["weight_map"].keys())
with open(CKPT) as f:
    ckpt_keys = set(json.load(f)["weight_map"].keys())

print(f"Base model keys: {len(base_keys)}")
print(f"Checkpoint keys: {len(ckpt_keys)}")
print(f"Keys in BOTH: {len(base_keys & ckpt_keys)}")
print(f"Keys ONLY in base: {len(base_keys - ckpt_keys)}")
print(f"Keys ONLY in ckpt: {len(ckpt_keys - base_keys)}")

if base_keys - ckpt_keys:
    print("\n--- Sample keys ONLY in base (first 10) ---")
    for k in sorted(base_keys - ckpt_keys)[:10]:
        print(f"  {k}")

if ckpt_keys - base_keys:
    print("\n--- Sample keys ONLY in ckpt (first 10) ---")
    for k in sorted(ckpt_keys - base_keys)[:10]:
        print(f"  {k}")

if base_keys & ckpt_keys:
    print("\n--- Sample MATCHING keys (first 5) ---")
    for k in sorted(base_keys & ckpt_keys)[:5]:
        print(f"  {k}")
