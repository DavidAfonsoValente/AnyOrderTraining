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

def get_masked_and_label_text(input_ids, labels, tokenizer):
    tokens = []
    label_tokens = []
    for tid, lid in zip(input_ids, labels):
        if tid == tokenizer.mask_token_id:
            tokens.append(f" [MASK({tokenizer.decode([lid])})] ")
        else:
            tokens.append(tokenizer.decode([tid]))
        
        if lid == -100:
            label_tokens.append("_")
        else:
            label_tokens.append(tokenizer.decode([lid]))
            
    return "".join(tokens), "".join(label_tokens)

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
    # For SFT methods, there are multiple examples per trajectory.
    # Let's check a few for the first trajectory.
    indices_to_check = [0, 1, 2] if len(dataset) > 2 else range(len(dataset))
    
    for i in indices_to_check:
        item = dataset[i]
        input_ids = item["input_ids"].tolist()
        labels = item["labels"].tolist()
        
        masked_text, label_text = get_masked_and_label_text(input_ids, labels, tokenizer)
        
        print(f"\nExample {i}:")
        # Find where the first [MASK] is
        mask_pos = masked_text.find("[MASK")
        if mask_pos != -1:
            start = max(0, mask_pos - 100)
            end = min(len(masked_text), mask_pos + 200)
            print(f"Context around MASK: ...{masked_text[start:end]}...")
        else:
            print(f"NO MASK FOUND! Input len: {len(input_ids)}")
            print(f"Input end: {masked_text[-200:]}")

        # Check labels for masked tokens
        masked_count = input_ids.count(tokenizer.mask_token_id)
        label_count = sum(1 for l in labels if l != -100)
        print(f"Masked count: {masked_count}, Label count: {label_count}")

methods = ["standard_sft", "prefix_sft_stage1", "prefix_sft_stage2", "aomt_mixed"]
for m in methods:
    verify_method(m)
