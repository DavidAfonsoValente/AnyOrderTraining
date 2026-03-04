#!/usr/bin/env python3
import sys
import os
import torch
import numpy as np

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from datasets import load_from_disk
from transformers import AutoTokenizer
from aomt.training.trainer import AOMTDataset
from aomt.training.mask_sampler import MaskMode

def verify_masking_strategies(data_path: str, model_name: str, num_examples: int = 2):
    """
    Loads the processed dataset and demonstrates the output of the AOMTDataset
    for each of the different masking strategies.

    Args:
        data_path (str): Path to the processed dataset directory.
        model_name (str): Name of the tokenizer model to use.
        num_examples (int): Number of examples to show for each strategy.
    """
    print("--- Verifying Masking Strategies ---")

    # 1. Load Tokenizer and Processed Data
    print(f"Loading tokenizer '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, low_cpu_mem_usage=True)
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({'mask_token': '[MASK]'})
        print("Added MASK token '[MASK]' to tokenizer.")

    print(f"Loading processed 'train' data from '{data_path}/train'...")
    processed_dataset = load_from_disk(os.path.join(data_path, "train"))
    print("Data loaded.")

    masking_strategies = [
        MaskMode.STANDARD_SFT,
        MaskMode.AOMT_ACTION_ONLY,
        MaskMode.AOMT_MIXED
    ]

    for strategy in masking_strategies:
        print(f"
========================================")
        print(f"  Verifying Strategy: {strategy.name}")
        print(f"========================================
")

        # 2. Create AOMTDataset for the specific strategy
        # For demonstration, mask_prob is set to 0.25
        mask_prob = 0.25 if strategy != MaskMode.STANDARD_SFT else 0.0
        
        dataset = AOMTDataset(
            processed_dataset=processed_dataset,
            tokenizer=tokenizer,
            mode=strategy,
            mask_prob=mask_prob
        )
        
        print(f"Displaying {num_examples} examples with mask_prob={mask_prob}:
")

        for i in range(num_examples):
            print(f"--- Example {i+1} for {strategy.name} ---")
            
            # 3. Get an item from the dataset
            item = dataset[i]
            input_ids = item['input_ids']
            target_ids = item['target_ids']
            loss_mask = item['loss_mask']
            
            # 4. Find the masked tokens
            masked_indices = torch.where(input_ids == tokenizer.mask_token_id)[0]
            loss_indices = torch.where(loss_mask == 1)[0]
            
            print(f"  Original Sequence Length: {len(target_ids)}")
            print(f"  Number of Masked Tokens: {len(masked_indices)}")
            print(f"  Number of Tokens in Loss Mask: {len(loss_indices)}")
            
            # 5. Decode and display a snippet of the text
            print("
  Decoded Snippet (showing effect of masking):")
            
            # Find a region with masking to display
            if len(masked_indices) > 0:
                start_idx = max(0, masked_indices[0] - 20)
                end_idx = min(len(input_ids), masked_indices[0] + 30)
            else: # If no masking, just show the start
                start_idx = 0
                end_idx = 50

            snippet_original = tokenizer.decode(target_ids[start_idx:end_idx])
            snippet_masked = tokenizer.decode(input_ids[start_idx:end_idx])

            print("
  ORIGINAL:")
            print(f"  ...{snippet_original}...")
            print("
  MASKED:")
            print(f"  ...{snippet_masked}...")
            print("
" + "-"*40)

if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    default_data_path = os.path.join(script_dir, '../data/processed_dataset')
    default_model_name = "inclusionAI/LLaDA2.0-mini"
    
    verify_masking_strategies(default_data_path, default_model_name)
