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
from tasks.train_standard_sft import apply_response_unit_mask
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
    tokenizer_path = config.get("model", {}).get("tokenizer_path", "models/LLaDA2.0-mini")
    train_path = config.get("data", {}).get("train_path")
    
    # 2. Load Tokenizer
    print(f"Loading tokenizer from '{tokenizer_path}'...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        # Fallback to LLaDA default if not set in tokenizer
        mask_token_id = 156895
    
    print(f"Using Mask Token ID: {mask_token_id} ({tokenizer.decode([mask_token_id])})")

    # 3. Initialize the correct Dataset based on the config
    print(f"Loading data from '{train_path}'...")
    
    if "aomt" in config:
        mode = config["aomt"]["mode"]
        mask_prob = config["aomt"]["mask_prob"]
        dataset = AOMTDataset(train_path, tokenizer, mask_prob, mode)
        
        print(f"\nDisplaying one AOMT example:\n")
        example_data = dataset[0]
        input_ids = example_data['input_ids']
        labels = example_data['labels']
        
        print("INPUT (with masks):")
        print(tokenizer.decode(input_ids, skip_special_tokens=False))
        print("\nTARGET (reconstructed from labels):")
        target_ids = input_ids.clone()
        mask = labels != -100
        target_ids[mask] = labels[mask]
        print(tokenizer.decode(target_ids, skip_special_tokens=False))
    else:
        # Standard SFT / Prefix SFT
        dataset = []
        with open(train_path, "r") as f:
            for i, line in enumerate(f):
                if i >= 5: break
                dataset.append(json.loads(line))
        
        print(f"\nDisplaying one SFT example (as seen by the model during training):\n")
        ex = dataset[0]
        messages = ex["messages"]
        
        # Tokenise full sequence
        input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt")[0]
        # Find prompt length
        prompt_text = tokenizer.apply_chat_template(messages[:1], add_generation_prompt=True, tokenize=False)
        prompt_len = len(tokenizer.encode(prompt_text))
        
        # Apply SFT masking logic
        masked_ids, labels = apply_response_unit_mask(input_ids.unsqueeze(0), torch.tensor([prompt_len]), mask_token_id)
        masked_ids = masked_ids[0]
        
        print("PROMPT (unmasked context):")
        print(tokenizer.decode(input_ids[:prompt_len], skip_special_tokens=False))
        print("\nRESPONSE (masked target):")
        print(tokenizer.decode(masked_ids[prompt_len:], skip_special_tokens=False))
        print("\nFULL INPUT TO MODEL:")
        print(tokenizer.decode(masked_ids, skip_special_tokens=False))

    print("\n" + "-"*40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize training data for a specific experiment config.")
    parser.add_argument("--config", type=str, required=True, help="Path to the experiment YAML config file.")
    
    args = parser.parse_args()
    verify_experiment_data(args.config)
