import torch
from transformers import AutoTokenizer
from aomt.data.dataset import AOMTDataset
from aomt.data.utils import load_robust_dataset
import os

def manual_verify():
    tokenizer = AutoTokenizer.from_pretrained("aomt/weights/LLaDA2.0-mini", trust_remote_code=True)
    ds_dict, _ = load_robust_dataset()
    raw_traj = [ds_dict["train"][0]] # Use the same trajectory for all
    
    methods = ["standard_sft", "prefix_sft_stage1", "aomt_mixed"]
    
    for method in methods:
        print(f"\n\n{'#'*30} VERIFYING METHOD: {method} {'#'*30}")
        ds = AOMTDataset(raw_traj, tokenizer, method=method, p_mask=0.25)
        item = ds[0]
        
        input_ids = item["input_ids"].tolist()
        labels = item["labels"].tolist()
        
        print(f"Sequence Length: {len(input_ids)}")
        
        # We want to see the "Boundary" between context and target
        # Target tokens are where labels != -100
        target_indices = [i for i, l in enumerate(labels) if l != -100]
        
        if not target_indices:
            print("ERROR: No target tokens found!")
            continue
            
        first_target = target_indices[0]
        last_target = target_indices[-1]
        
        # 1. Print the context just before the target
        context_before = input_ids[max(0, first_target-10) : first_target]
        print(f"\nContext before target: '{tokenizer.decode(context_before)}'")
        
        # 2. Print the target content
        target_content = labels[first_target : last_target+1]
        print(f"Target content (masked): '{tokenizer.decode(target_content)}'")
        
        # 3. Print the trailing tokens
        context_after = input_ids[last_target+1 : last_target+10]
        if context_after:
            print(f"Context after target: '{tokenizer.decode(context_after)}'")
        else:
            print("Context after target: [END OF SEQUENCE]")

        # 4. Verify Role Markers
        full_text = tokenizer.decode(input_ids)
        has_human_start = "<role>HUMAN</role>" in full_text or "HUMAN" in full_text
        print(f"\nRole Check: Start Marker Present? {has_human_start}")
        
        # 5. Verify Mask Token
        is_masked = all(input_ids[i] == tokenizer.mask_token_id for i in target_indices)
        print(f"Mask Check: All target tokens replaced by MASK? {is_masked}")

if __name__ == "__main__":
    manual_verify()
