import torch
from transformers import AutoTokenizer
from aomt.data.dataset import AOMTDataset
from aomt.data.utils import load_robust_dataset
import os

def deep_audit():
    tokenizer = AutoTokenizer.from_pretrained("aomt/weights/LLaDA2.0-mini", trust_remote_code=True)
    mask_id = tokenizer.mask_token_id
    
    ds_dict, _ = load_robust_dataset()
    train_ds = ds_dict["train"]
    
    # 1. Find a REAL long trajectory (not a placeholder)
    real_example = None
    for i in range(len(train_ds)):
        convs = train_ds[i]["conversations"]
        if len(convs) > 6: # Find one with at least 3-4 steps
            real_example = [train_ds[i]]
            break
            
    if not real_example:
        print("Could not find a long trajectory. Using first available.")
        real_example = [train_ds[0]]

    # 2. Audit Main Methods
    for method in ["standard_sft", "aomt_mixed"]:
        print(f"\n\n{'#'*30} AUDIT: {method.upper()} {'#'*30}")
        # Use p_mask=0.2 for clear visualization
        ds = AOMTDataset(real_example, tokenizer, method=method, p_mask=0.2)
        item = ds[0]
        
        ids = item["input_ids"].tolist()
        labels = item["labels"].tolist()
        
        # --- BUILD INPUT VIEW ---
        input_view = []
        for i, tid in enumerate(ids):
            if tid == mask_id:
                input_view.append("[MASK]")
            else:
                input_view.append(tokenizer.decode([tid]))
        
        # --- BUILD TARGET VIEW ---
        target_view = []
        for i, lid in enumerate(labels):
            if lid == -100:
                target_view.append(" _ ") # Context token
            else:
                target_view.append(tokenizer.decode([lid]))
        
        print("\n[STEP 1] WHAT THE MODEL SEES (INPUT):")
        print("-" * 50)
        print("".join(input_view))
        
        print("\n[STEP 2] WHAT THE MODEL PREDICTS (TARGETS):")
        print("-" * 50)
        print("".join(target_view))
        
        print("\n[STEP 3] DISTRIBUTION CHECK:")
        full_text = "".join(input_view)
        inside_user = "<role>HUMAN</role>" in full_text and "<|role_end|>" in full_text
        print(f"Is generating inside USER role? {inside_user}")
        
        # Check if the last role marker is after the masks (Standard SFT only)
        if method == "standard_sft":
            last_mask_idx = max([i for i, tid in enumerate(ids) if tid == mask_id])
            role_end_ids = tokenizer.encode("<|role_end|>", add_special_tokens=False)
            # Find index of <|role_end|>
            # In LLaDA it is usually the very last token
            is_at_end = ids[-1] in [156900, 157152] # Common role end IDs
            print(f"Does sequence end with <|role_end|>? {is_at_end}")

if __name__ == "__main__":
    deep_audit()
