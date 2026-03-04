#!/usr/bin/env python3
import sys
import os
import torch
import numpy as np
import yaml
import argparse

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from datasets import load_from_disk
from transformers import AutoTokenizer
from aomt.training.trainer import AOMTDataset
from aomt.training.mask_sampler import MaskMode

def verify_experiment_data(config_path: str, data_path: str, num_examples: int = 2):
    """
    Loads a specific experiment configuration and visualizes how the data
    will be masked for that experiment.

    Args:
        config_path (str): Path to the YAML experiment config file.
        data_path (str): Path to the processed dataset directory.
        num_examples (int): Number of examples to show.
    """
    print(f"
========================================")
    print(f"  Verifying Experiment Config: {os.path.basename(config_path)}")
    print(f"========================================
")

    # 1. Load Config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_name = config.get("model", "inclusionAI/LLaDA2.0-mini")
    mask_mode_str = config.get("mask_mode", "standard_sft")
    mask_prob = config.get("mask_prob", 0.0)
    
    try:
        mask_mode = MaskMode(mask_mode_str)
    except ValueError:
        print(f"Error: Invalid mask_mode '{mask_mode_str}' in config.")
        return

    # 2. Load Tokenizer and Processed Data
    print(f"Loading tokenizer '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, low_cpu_mem_usage=True)
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({'mask_token': '[MASK]'})
        print("Added MASK token '[MASK]' to tokenizer.")

    print(f"Loading processed 'train' data from '{data_path}/train'...")
    processed_dataset = load_from_disk(os.path.join(data_path, "train"))
    print("Data loaded.")

    # 3. Create AOMTDataset for the specific strategy
    dataset = AOMTDataset(
        processed_dataset=processed_dataset,
        tokenizer=tokenizer,
        mode=mask_mode,
        mask_prob=mask_prob if mask_prob is not None else 0.0
    )
    
    print(f"Displaying {num_examples} examples with mask_prob={dataset.mask_prob}:
")

    for i in range(num_examples):
        print(f"--- Example {i+1} for {mask_mode.name} ---")
        
        item = dataset[i]
        input_ids = item['input_ids']
        target_ids = item['target_ids']
        loss_mask = item['loss_mask']
        
        masked_indices = torch.where(input_ids == tokenizer.mask_token_id)[0]
        loss_indices = torch.where(loss_mask == 1)[0]
        
        print(f"  Original Sequence Length: {len(target_ids)}")
        print(f"  Number of Masked Tokens: {len(masked_indices)}")
        print(f"  Number of Tokens in Loss Mask: {len(loss_indices)}")
        
        print("\n  Decoded Snippet (showing effect of masking):")
        
        if len(masked_indices) > 0:
            start_idx = max(0, masked_indices[0] - 30)
            end_idx = min(len(input_ids), masked_indices[-1] + 30)
        else:
            start_idx = 0
            end_idx = 60

        snippet_original = tokenizer.decode(target_ids[start_idx:end_idx])
        snippet_masked = tokenizer.decode(input_ids[start_idx:end_idx])

        print("\n  ORIGINAL:")
        print(f"  ...{snippet_original}...")
        print("\n  MASKED:")
        print(f"  ...{snippet_masked}...")
        print("\n" + "-"*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize training data for a specific experiment config.")
    parser.add_argument("--config", type=str, required=True, help="Path to the experiment YAML config file.")
    
    script_dir = os.path.dirname(__file__)
    default_data_path = os.path.join(script_dir, '../data/processed_dataset')
    
    args = parser.parse_args()
    
    verify_experiment_data(args.config, default_data_path)
