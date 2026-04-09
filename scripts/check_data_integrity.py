import torch
from transformers import AutoTokenizer
from aomt.data.dataset import AOMTDataset
from aomt.data.utils import load_robust_dataset
import os

def visualize_masks():
    tokenizer = AutoTokenizer.from_pretrained("aomt/weights/LLaDA2.0-mini", trust_remote_code=True)
    mask_token_id = tokenizer.mask_token_id # 156895
    
    ds_dict, _ = load_robust_dataset()
    raw_traj = [ds_dict["train"][0]]
    
    methods = ["standard_sft", "prefix_sft_stage1", "aomt_mixed"]
    
    for method in methods:
        print(f"\n\n{'='*40} {method.upper()} {'='*40}")
        ds = AOMTDataset(raw_traj, tokenizer, method=method, p_mask=0.4) # Use 40% mask for mixed to see it clearly
        item = ds[0]
        
        input_ids = item["input_ids"].tolist()
        labels = item["labels"].tolist()
        
        # We will decode token by token to show exactly where masks are
        decoded_parts = []
        for i, token_id in enumerate(input_ids):
            if token_id == mask_token_id:
                # This is a mask! Show the target it's hiding in brackets
                target_text = tokenizer.decode([labels[i]]) if labels[i] != -100 else "???"
                decoded_parts.append(f"[{target_text.strip() if target_text.strip() else 'SPC'}_MASK]")
            else:
                decoded_parts.append(tokenizer.decode([token_id]))
        
        full_visualization = "".join(decoded_parts)
        print("MODEL INPUT VISUALIZATION:")
        print("-" * 80)
        print(full_visualization)
        print("-" * 80)
        
        # Sanity Check
        mask_count = input_ids.count(mask_token_id)
        label_count = sum(1 for l in labels if l != -100)
        print(f"Stats: {mask_count} masks vs {label_count} labels. Match? {mask_count == label_count}")

if __name__ == "__main__":
    visualize_masks()
