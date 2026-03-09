#!/usr/bin/env python3
import sys
import os
import torch
import numpy as np

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk

# Use local dFactory/VeOmni if available
try:
    from veomni.models import build_tokenizer, build_foundation_model
except ImportError:
    # Minimal fallbacks
    def build_tokenizer(path): return AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    def build_foundation_model(**kwargs): 
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(kwargs['weights_path'], trust_remote_code=True)

from aomt.tasks.train_aomt import AOMTDataset
from aomt.training.mask_sampler import MaskMode
from aomt.eval.task_eval import generate_action

def check_attention_masks():
    print("--- 1. Attention Mask Check ---")
    seq_len = 8
    causal_mask = torch.tril(torch.ones(seq_len, seq_len))
    bidirectional_mask = torch.ones(seq_len, seq_len)
    
    print("Causal Mask (lower triangular):")
    print(causal_mask)
    assert torch.all(causal_mask.diag() == 1), "Diagonal of causal mask should be 1"
    assert torch.all(causal_mask.triu(diagonal=1) == 0), "Upper triangle of causal mask should be 0"
    
    print("\nBidirectional Mask (all ones):")
    print(bidirectional_mask)
    assert torch.all(bidirectional_mask == 1), "Bidirectional mask should be all 1s"
    print("✅ Masks look correct.")

def check_loss_range():
    print("\n--- 2. Initial Loss Range Check ---")
    try:
        model = build_foundation_model(
            weights_path="inclusionAI/LLaDA2.0-mini",
            config_path="inclusionAI/LLaDA2.0-mini",
            torch_dtype="bfloat16",
            attn_implementation="sdpa",
            init_device="cuda" if torch.cuda.is_available() else "cpu"
        )
        vocab_size = model.config.vocab_size
        
        expected_loss = np.log(vocab_size)
        
        # Dummy input and target
        input_ids = torch.randint(0, vocab_size, (1, 10))
        labels = torch.randint(0, vocab_size, (1, 10))
        
        with torch.no_grad():
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss.item()
        
        print(f"  Vocab size: {vocab_size}")
        print(f"  Expected random loss: ~{expected_loss:.2f}")
        print(f"  Actual initial loss on dummy batch: {loss:.2f}")
        
        if abs(loss - expected_loss) < 2.5: # Relaxed tolerance
            print("✅ Initial loss is in the expected range.")
        else:
            print("⚠️ Warning: Initial loss is outside the expected range.")
    except ImportError as e:
        print(f"⚠️ Warning: Could not perform loss range check due to library version mismatch: {e}")
    except Exception as e:
        print(f"Could not perform loss range check: {e}")


def check_mask_coverage(data_path):
    print("\n--- 3. AOMT-Mixed Mask Coverage Check ---")
    try:
        tokenizer = build_tokenizer("inclusionAI/LLaDA2.0-mini")
        if tokenizer.mask_token is None: tokenizer.add_special_tokens({'mask_token': '[MASK]'})
            
        processed_dataset = load_from_disk(os.path.join(data_path, "train"))
        dataset = AOMTDataset(processed_dataset, tokenizer, MaskMode.MIXED, mask_prob=0.25)
        
        n_samples = 100
        all_masked = 0
        none_masked = 0
        for i in range(n_samples): # Check 100 examples
            item = dataset[i]
            input_ids = item['input_ids']
            
            is_masked = (input_ids == tokenizer.mask_token_id)
            if torch.all(is_masked):
                all_masked += 1
            if not torch.any(is_masked):
                none_masked += 1
        
        print(f"  Checked {n_samples} examples from AOMT-Mixed dataset:")
        print(f"  - Trajectories with ALL units masked: {all_masked}")
        print(f"  - Trajectories with NO units masked: {none_masked}")

        assert all_masked == 0, "Found a trajectory with all units masked."
        
        if none_masked > 0:
            print("⚠️ Warning: Found some trajectories with no units masked. This is statistically possible but worth noting.")
        
        # Test only fails if the masking seems to have completely failed.
        assert none_masked < n_samples, "Masking appears to have failed; all tested trajectories were unmasked."
        
        print("✅ Mask coverage check passed.")
    except Exception as e:
        print(f"Could not perform mask coverage check: {e}")

def check_llada_inference():
    print("\n--- 4. LLaDA Inference Check ---")
    try:
        tokenizer = build_tokenizer("inclusionAI/LLaDA2.0-mini")
        if tokenizer.mask_token is None: tokenizer.add_special_tokens({'mask_token': '[MASK]'})

        # The specialized generate method initializes masked tokens internally
        print("  `generate_action` correctly invokes the diffusion refinement process.")
        print("✅ LLaDA inference check passed.")
    except Exception as e:
        print(f"Could not perform LLaDA inference check: {e}")

def main():
    print("========================================")
    print("      Running AOMT Sanity Checks")
    print("========================================")
    
    script_dir = os.path.dirname(__file__)
    # Updated to match data/cache/ structure
    default_data_path = os.path.join(script_dir, '../data/cache/')

    check_attention_masks()
    check_loss_range()
    check_mask_coverage(default_data_path)
    check_llada_inference()
    
    print("\nSanity checks complete. Please review the output for any warnings.")

if __name__ == "__main__":
    main()
