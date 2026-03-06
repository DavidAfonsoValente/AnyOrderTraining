#!/usr/bin/env python3
import sys
import os
import torch
import numpy as np
import yaml
import argparse

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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

def verify_experiment_data(config_path: str, data_path: str):
    """
    Loads a specific experiment configuration and visualizes one example from each
    of the core environments (alfworld, scienceworld, webshop).
    """
    print(f"\n========================================")
    print(f"  Verifying Experiment Config: {os.path.basename(config_path)}")
    print(f"========================================\n")

    # 1. Load Config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Use the local model path from the updated configs
    model_path = config.get("model", {}).get("tokenizer_path", "models/LLaDA2.0-mini")
    mask_mode_str = config.get("mask_mode", "standard_sft")
    
    try:
        mask_mode = MaskMode(mask_mode_str)
    except ValueError:
        print(f"Error: Invalid mask_mode '{mask_mode_str}' in config.")
        return

    # 2. Load Tokenizer and Processed Data
    print(f"Loading tokenizer from '{model_path}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, low_cpu_mem_usage=True)
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({'mask_token': '[MASK]'})

    print(f"Loading processed 'train' data from '{data_path}/train'...")
    processed_dataset = load_from_disk(os.path.join(data_path, "train"))
    print("Data loaded.")

    # 3. Initialize the correct Dataset based on the mask_mode
    if mask_mode == MaskMode.PREFIX_SFT_STAGE1:
        dataset = SummarizedTrajectoryDataset(processed_dataset, tokenizer, build_prefix_sft_examples)
    elif mask_mode == MaskMode.PREFIX_SFT_STAGE2:
        dataset = SummarizedTrajectoryDataset(processed_dataset, tokenizer, build_prefix_sft_stage2_examples)
    elif mask_mode == MaskMode.STANDARD_SFT:
        dataset = SummarizedTrajectoryDataset(processed_dataset, tokenizer, build_standard_sft_examples)
    else: # For MIXED, ACTION_ONLY, etc.
        dataset = AOMTDataset(processed_dataset, tokenizer, mode=mask_mode, mask_prob=config.get("mask_prob", 0.0))

    if not dataset:
        print("\nWarning: No examples were generated for this configuration. Cannot display anything.")
        return

    print(f"\nSearching for one example from each environment for mode '{mask_mode.name}':\n")
    
    # 4. Find and display one example per environment
    target_envs = {"alfworld", "scienceworld", "webshop"}
    found_envs = set()
    is_summarized = isinstance(dataset, SummarizedTrajectoryDataset)

    for i in range(len(dataset)):
        # Stop if we've found one of each
        if len(found_envs) == len(target_envs):
            break

        example_data = dataset[i]
        
        # Determine the environment for the current example
        env = "unknown"
        if is_summarized:
            # Summarized examples have 'env' in their dictionary
            env = example_data.get('env', 'unknown')
        else:
            # AOMTDataset needs to look up the env from the original processed_dataset
            env = dataset.processed_dataset[i]['env']

        if env in target_envs and env not in found_envs:
            print(f"--- Example for env='{env}' ---")
            
            input_ids = example_data['input_ids']
            target_ids = example_data['target_ids']
            loss_mask = example_data['loss_mask']
            
            masked_indices = torch.where(input_ids == tokenizer.mask_token_id)[0]
            
            print(f"  Sequence Length: {len(target_ids)}")
            print(f"  Number of Masked Tokens: {len(masked_indices)}")
            
            print("\n  Decoded Example:")
            
            # For summarized datasets or short sequences, show the whole thing
            if is_summarized or len(input_ids) < 300:
                snippet_original = tokenizer.decode(target_ids, skip_special_tokens=False)
                snippet_masked = tokenizer.decode(input_ids, skip_special_tokens=False)
                print("\n  INPUT (masked):")
                print(f"  {snippet_masked}")
                print("\n  TARGET (ground truth):")
                print(f"  {snippet_original}")
            else: # For long trajectories, show a snippet around the first mask
                if len(masked_indices) > 0:
                    start_idx = max(0, masked_indices[0] - 100)
                    end_idx = min(len(input_ids), masked_indices[0] + 100)
                else: # No masks found, just show the beginning
                    start_idx, end_idx = 0, 200

                snippet_original = tokenizer.decode(target_ids[start_idx:end_idx], skip_special_tokens=True)
                snippet_masked = tokenizer.decode(input_ids[start_idx:end_idx], skip_special_tokens=True)

                print("\n  INPUT (masked snippet):")
                print(f"  ...{snippet_masked}...")
                print("\n  TARGET (ground truth snippet):")
                print(f"  ...{snippet_original}...")

            found_envs.add(env)
            print("\n" + "-"*40)

    if not found_envs:
        print("Could not find any examples for the target environments in the dataset.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize training data for a specific experiment config.")
    parser.add_argument("--config", type=str, required=True, help="Path to the experiment YAML config file.")
    
    # Correctly locate the data directory relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_path = os.path.join(script_dir, '../data/processed_dataset')
    
    parser.add_argument("--data_path", type=str, default=default_data_path, help="Path to the processed dataset directory.")
    
    args = parser.parse_args()
    
    verify_experiment_data(args.config, args.data_path)
