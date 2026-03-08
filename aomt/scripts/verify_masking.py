#!/usr/bin/env python3
import sys
import os
import torch
import numpy as np
import yaml
import argparse
import json

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets import load_from_disk
from transformers import AutoTokenizer
from tqdm import tqdm

# Import all the necessary components from the training pipeline
from tasks.train_aomt import apply_unit_mask, AOMTDataset
from training.mask_sampler import MaskMode

def verify_experiment_data(config_path: str):
    """
    Loads a specific experiment configuration and visualizes one example.
    """
    print(f"\n========================================")
    print(f"  Verifying Experiment Config: {os.path.basename(config_path)}")
    print(f"========================================\n")

    # 1. Load Config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Use the local model path from the updated configs
    tokenizer_path = config.get("model", {}).get("tokenizer_path", "models/llada2-mini-sep")
    train_path = config.get("data", {}).get("train_path")
    
    # 2. Load Tokenizer
    print(f"Loading tokenizer from '{tokenizer_path}'...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    # 3. Initialize the correct Dataset based on the config
    print(f"Loading data from '{train_path}'...")
    
    if "aomt" in config:
        mode = config["aomt"]["mode"]
        mask_prob = config["aomt"]["mask_prob"]
        dataset = AOMTDataset(train_path, tokenizer, mask_prob, mode)
    else:
        # Standard SFT / Prefix SFT
        # We can use a simple reader for visualization
        dataset = []
        with open(train_path, "r") as f:
            for i, line in enumerate(f):
                if i >= 5: break # Just a few for viz
                dataset.append(json.loads(line))
        
    print(f"\nDisplaying one example:\n")
    
    if "aomt" in config:
        example_data = dataset[0]
        input_ids = example_data['input_ids']
        labels = example_data['labels']
        
        print("INPUT (masked):")
        print(tokenizer.decode(input_ids, skip_special_tokens=False))
        print("\nTARGET (labels):")
        # Decode only the masked parts
        target_ids = input_ids.clone()
        mask = labels != -100
        target_ids[mask] = labels[mask]
        print(tokenizer.decode(target_ids, skip_special_tokens=False))
    else:
        # Chat format
        ex = dataset[0]
        print("PROMPT:")
        print(ex["messages"][0]["content"])
        print("\nRESPONSE:")
        print(ex["messages"][1]["content"])

    print("\n" + "-"*40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize training data for a specific experiment config.")
    parser.add_argument("--config", type=str, required=True, help="Path to the experiment YAML config file.")
    
    args = parser.parse_args()
    verify_experiment_data(args.config)
