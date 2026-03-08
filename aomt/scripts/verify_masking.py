#!/usr/bin/env python3
import sys
import os
import torch
import numpy as np
import yaml
import argparse
import json

# Add the project root and dFactory to the Python path
AOMT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, AOMT_ROOT)
sys.path.insert(0, os.path.join(AOMT_ROOT, 'dFactory'))
sys.path.insert(0, os.path.join(AOMT_ROOT, 'dFactory', 'VeOmni'))

from transformers import AutoTokenizer

# Import all the necessary components from the training pipeline
from tasks.train_aomt import apply_unit_mask, AOMTDataset
from tasks.train_standard_sft import apply_response_unit_mask

def colorize_text(text, is_masked):
    """Simple terminal coloring for visualization."""
    if is_masked:
        return f"\033[91m{text}\033[0m" # Red for masked
    else:
        return f"\033[92m{text}\033[0m" # Green for context

def verify_experiment_data(config_path: str, num_examples: int = 3):
    """
    Loads a specific experiment configuration and visualizes examples.
    """
    print(f"\n" + "="*80)
    print(f"  Verifying Experiment Config: {os.path.basename(config_path)}")
    print("="*80 + "\n")

    # 1. Load Config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    tokenizer_path = config.get("model", {}).get("tokenizer_path", "weights/LLaDA2.0-mini")
    train_path = config.get("data", {}).get("train_path")
    
    # 2. Load Tokenizer
    print(f"Loading tokenizer from '{tokenizer_path}'...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    mask_token_id = tokenizer.mask_token_id or 156895
    print(f"Using Mask Token ID: {mask_token_id}")

    # 3. Initialize Dataset
    print(f"Loading data from '{train_path}'...")
    
    if "aomt" in config:
        mode = config["aomt"]["mode"]
        mask_prob = config["aomt"]["mask_prob"]
        dataset = AOMTDataset(train_path, tokenizer, mask_prob, mode, mask_token_id)
        
        for i in range(min(num_examples, len(dataset))):
            print(f"\n--- Example {i+1} (AOMT {mode}) ---")
            example_data = dataset[i]
            input_ids = example_data['input_ids']
            labels = example_data['labels']

            num_tokens = len(input_ids)
            mask = (labels != -100)
            num_masked = mask.sum().item()
            
            print(f"Stats: Tokens={num_tokens}, Masked={num_masked} ({num_masked/num_tokens:.1%})")
            
            # Reconstruct colored string
            colored_output = ""
            for tid, is_m in zip(input_ids, mask):
                token_text = tokenizer.decode([tid], skip_special_tokens=False)
                if is_m:
                    # For masked positions, tid is mask_token_id, we want the label
                    true_token = tokenizer.decode([labels[mask][ (labels[mask] == tid).nonzero(as_tuple=True)[0][0] if (labels[mask] == tid).any() else 0 ]], skip_special_tokens=False)
                    # Simple hack: just show [MASK] for masked, or the target if we can align easily
                    # Better: decode label at that position
                    true_tid = labels[ (input_ids == tid).nonzero(as_tuple=True)[0][0] ] # This is buggy logic
            
            # Correct approach: iterate by index
            tokens_str = []
            for j in range(num_tokens):
                tid = input_ids[j]
                is_m = mask[j]
                if is_m:
                    actual_tid = labels[j]
                    text = tokenizer.decode([actual_tid], skip_special_tokens=False)
                    tokens_str.append(colorize_text(text, True))
                else:
                    text = tokenizer.decode([tid], skip_special_tokens=False)
                    tokens_str.append(colorize_text(text, False))
            
            print("Visual (Red=Masked, Green=Context):")
            print("".join(tokens_str))
            
            # Logical check
            if mode == "action_only":
                # Verify no HUMAN turns are masked
                full_text = tokenizer.decode(input_ids, skip_special_tokens=False)
                if "\033[91m<role>HUMAN</role>" in "".join(tokens_str):
                    print("❌ FAIL: HUMAN role tag masked in action_only mode")
                else:
                    print("✅ PASS: Role integrity maintained")

    else:
        # Standard SFT / Prefix SFT
        from tasks.train_standard_sft import SFTDataset
        dataset = SFTDataset(train_path, tokenizer, config["train"]["max_seq_length"])
        
        for i in range(min(num_examples, len(dataset))):
            print(f"\n--- Example {i+1} (SFT/Prefix) ---")
            item = dataset[i]
            input_ids = item["input_ids"]
            prompt_len = item["prompt_len"]
            
            masked_ids, labels = apply_response_unit_mask(input_ids.unsqueeze(0), torch.tensor([prompt_len]), mask_token_id)
            masked_ids = masked_ids[0]
            labels = labels[0]
            
            num_tokens = len(masked_ids)
            mask = (labels != -100)
            num_masked = mask.sum().item()
            print(f"Stats: Tokens={num_tokens}, Prompt={prompt_len}, Masked={num_masked}")
            
            tokens_str = []
            for j in range(len(input_ids)):
                tid = masked_ids[j]
                is_m = mask[j]
                if is_m:
                    text = tokenizer.decode([input_ids[j]], skip_special_tokens=False)
                    tokens_str.append(colorize_text(text, True))
                else:
                    text = tokenizer.decode([tid], skip_special_tokens=False)
                    tokens_str.append(colorize_text(text, False))
            
            print("Visual:")
            print("".join(tokens_str))
            
            # Validation
            if (mask[:prompt_len]).any():
                print("❌ FAIL: Masking found in prompt region")
            elif not (mask[prompt_len:]).all() and prompt_len < len(input_ids):
                # For SFT, everything after prompt should be masked
                print("❌ FAIL: Response region not fully masked")
            else:
                print("✅ PASS: SFT masking boundaries correct")

    print("\n" + "-"*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize training data.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--num", type=int, default=1)
    args = parser.parse_args()
    verify_experiment_data(args.config, args.num)
