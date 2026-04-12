import sys
import json
import torch
from transformers import AutoTokenizer

# Add project root to path
sys.path.append("/home/davidvalente/AnyOrderTraining")

from aomt.data.tokenize_trajectory import tokenize_trajectory, _ensure_list_of_ints
from aomt.data.masking import (
    apply_unit_mask,
    sample_sft_mask,
    sample_prefix_stage1_mask,
    sample_aomt_mixed_mask
)
from aomt.data.dataset import AOMTDataset

model_path = "/home/davidvalente/AnyOrderTraining/aomt/weights/LLaDA2.0-mini"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

with open("/home/davidvalente/AnyOrderTraining/samples.json", "r") as f:
    samples = json.load(f)

def verify_method(method, p_mask=0.25):
    print(f"\n{'='*20} VERIFYING METHOD: {method} {'='*20}")
    dataset = AOMTDataset(
        raw_dataset=samples,
        tokenizer=tokenizer,
        method=method,
        p_mask=p_mask,
        cache_dir="data/cache_verify",
        split="verify"
    )
    
    # Check a few items from the dataset
    num_to_check = min(3, len(dataset))
    for i in range(num_to_check):
        item = dataset[i]
        input_ids = item["input_ids"].tolist()
        labels = item["labels"].tolist()
        
        decoded_input = tokenizer.decode([tid if tid != tokenizer.mask_token_id else 0 for tid in input_ids]) # replace mask with 0 for decoding
        # More robust decoding: replace mask with special string
        decoded_tokens = []
        for tid in input_ids:
            if tid == tokenizer.mask_token_id:
                decoded_tokens.append("[MASK]")
            else:
                decoded_tokens.append(tokenizer.decode([tid]))
        
        decoded_input_str = "".join(decoded_tokens)
        
        # Decode labels
        label_tokens = []
        for tid in labels:
            if tid == -100:
                label_tokens.append("_")
            else:
                label_tokens.append(tokenizer.decode([tid]))
        decoded_labels_str = "".join(label_tokens)
        
        print(f"\nExample {i}:")
        print(f"Input: {decoded_input_str[:200]}...")
        print(f"Labels: {decoded_labels_str[:200]}...")
        
        # Verification logic
        if method == "standard_sft":
            # Current implementation: masks one action at step t.
            # User wants: "Every action turn is masked".
            pass
        elif method == "prefix_sft_stage1":
            # Mask next observation
            pass
        elif method == "prefix_sft_stage2":
            # Mask current action
            pass
        elif method == "aomt_mixed":
            # Random masking
            mask_count = input_ids.count(tokenizer.mask_token_id)
            print(f"Masked tokens: {mask_count} / {len(input_ids)}")

methods = ["standard_sft", "prefix_sft_stage1", "prefix_sft_stage2", "aomt_mixed"]
for m in methods:
    verify_method(m)
