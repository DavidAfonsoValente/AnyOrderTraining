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
from tqdm import tqdm

# Import all the necessary components from the training pipeline
from aomt.training.trainer import AOMTDataset, SummarizedTrajectoryDataset
from aomt.training.mask_sampler import MaskMode
from aomt.training.collator import (
    build_prefix_sft_examples,
    build_prefix_sft_stage2_examples,
    build_standard_sft_examples
)

def verify_experiment_data(config_path: str, data_path: str, num_examples: int = 2):
    """
    Loads a specific experiment configuration and visualizes how the data
    will be masked for that experiment, now handling all dataset types.

    Args:
        config_path (str): Path to the YAML experiment config file.
        data_path (str): Path to the processed dataset directory.
        num_examples (int): Number of examples to show.
    """
    print(f"\n========================================")
    print(f"  Verifying Experiment Config: {os.path.basename(config_path)}")
    print(f"========================================\n")

    # 1. Load Config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_name = config.get("model", "inclusionAI/LLaDA2.0-mini")
    mask_mode_str = config.get("mask_mode", "standard_sft")
    
    try:
        mask_mode = MaskMode(mask_mode_str)
    except ValueError:
        print(f"Error: Invalid mask_mode '{mask_mode_str}' in config.")
        return

    # 2. Load Tokenizer and Processed Data
    print(f"Loading tokenizer '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, low_cpu_mem_usage=True)
    if tokenizer.mask_token is None:
        # Add mask token if it doesn't exist. For LLaMA, it's often not set.
        tokenizer.add_special_tokens({'mask_token': '[MASK]'})
        print("Added MASK token '[MASK]' to tokenizer.")

    print(f"Loading processed 'train' data from '{data_path}/train'...")
    processed_dataset = load_from_disk(os.path.join(data_path, "train"))
    print("Data loaded.")

    # 3. Initialize the correct Dataset based on the mask_mode
    # This logic now mirrors the main training script.
    if mask_mode == MaskMode.PREFIX_SFT_STAGE1:
        dataset = SummarizedTrajectoryDataset(processed_dataset, tokenizer, build_prefix_sft_examples)
    elif mask_mode == MaskMode.PREFIX_SFT_STAGE2:
        dataset = SummarizedTrajectoryDataset(processed_dataset, tokenizer, build_prefix_sft_stage2_examples)
    elif mask_mode == MaskMode.STANDARD_SFT:
        dataset = SummarizedTrajectoryDataset(processed_dataset, tokenizer, build_standard_sft_examples)
    else: # For MIXED, ACTION_ONLY, etc.
        dataset = AOMTDataset(processed_dataset, tokenizer, mode=mask_mode, mask_prob=config.get("mask_prob", 0.0))

    # For summarized datasets, num_examples might be larger than the dataset if a trajectory yields few examples
    actual_num_examples = min(num_examples, len(dataset))
    if actual_num_examples == 0:
        print("\nWarning: No examples were generated for this configuration. Cannot display anything.")
        print("-" * 40)
        return

    print(f"Displaying {actual_num_examples} examples for mode '{mask_mode.name}':\n")

    for i in range(actual_num_examples):
        print(f"--- Example {i+1} for {mask_mode.name} ---")
        
        # __getitem__ returns a dict with pre-masked inputs for all dataset types now
        item = dataset[i]
        input_ids = item['input_ids']
        target_ids = item['target_ids']
        loss_mask = item['loss_mask']
        
        masked_indices = torch.where(input_ids == tokenizer.mask_token_id)[0]
        loss_indices = torch.where(loss_mask == 1)[0]
        
        print(f"  Sequence Length: {len(target_ids)}")
        print(f"  Number of Masked Tokens: {len(masked_indices)}")
        print(f"  Number of Tokens in Loss Mask: {len(loss_indices)}")
        
        print("\n  Decoded Example:")
        
        # The entire sequence is usually short for summarized datasets, so we show it all
        snippet_original = tokenizer.decode(target_ids)
        snippet_masked = tokenizer.decode(input_ids)

        print("\n  CONTEXT (Input to be Unmasked):")
        print(f"  ...{snippet_masked}...")
        print("\n  TARGET (Ground Truth):")
        print(f"  ...{snippet_original}...")
        print("\n" + "-"*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize training data for a specific experiment config.")
    parser.add_argument("--config", type=str, required=True, help="Path to the experiment YAML config file.")
    
    script_dir = os.path.dirname(__file__)
    default_data_path = os.path.join(script_dir, '../data/processed_dataset')
    
    args = parser.parse_args()
    
    verify_experiment_data(args.config, default_data_path)
