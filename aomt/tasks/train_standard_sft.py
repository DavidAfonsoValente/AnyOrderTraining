"""
tasks/train_standard_sft.py
Baseline custom task for Standard SFT, Prefix SFT Stage 1, and Stage 2.
Verified implementation based on Engineering Specification v3.
"""

import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_scheduler
import yaml
import argparse
from tqdm import tqdm
import numpy as np

# Use local dFactory/VeOmni if available, otherwise fallback to local trainer logic
try:
    from veomni.models import build_tokenizer, build_foundation_model
    from veomni.distributed.torch_parallelize import build_parallelize_model
    # Import other necessary dFactory components
except ImportError:
    # Fallback/Mock for local testing if veomni is not in path
    def build_tokenizer(path): return AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    def build_foundation_model(**kwargs): 
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(kwargs['weights_path'], trust_remote_code=True)

# ---- Spec Section 6: Masking logic ------------------------------------------

def apply_response_unit_mask(input_ids: torch.Tensor,
                              prompt_lengths: torch.Tensor,
                              mask_token_id: int) -> tuple:
    """
    Deterministically masks the entire response span.
    """
    B, L = input_ids.shape
    device = input_ids.device
    positions = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
    response_mask = positions >= prompt_lengths.unsqueeze(1)

    masked_input_ids = input_ids.clone()
    masked_input_ids[response_mask] = mask_token_id

    labels = torch.full_like(input_ids, -100)
    labels[response_mask] = input_ids[response_mask]

    return masked_input_ids, labels


def compute_unit_mask_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over masked positions only. No 1/p_mask weighting."""
    if (labels == -100).all():
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
    )

# ---- Dataset and Collator ---------------------------------------------------

class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, jsonl_path, tokenizer, max_seq_length):
        self.examples = []
        with open(jsonl_path, "r") as f:
            for line in f:
                self.examples.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        messages = ex["messages"]
        
        # Tokenize using apply_chat_template for consistency with LLaDA/dFactory
        input_ids = self.tokenizer.apply_chat_template(messages, tokenize=True, truncation=True, max_length=self.max_seq_length)
        
        # Determine prompt length by tokenizing messages[:-1] with generation prompt
        prompt_str = self.tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
        prompt_ids = self.tokenizer(prompt_str, add_special_tokens=False)["input_ids"]
        prompt_len = min(len(prompt_ids), len(input_ids))
        
        return {"input_ids": torch.tensor(input_ids), "prompt_len": prompt_len}

def collate_fn(batch, pad_token_id):
    input_ids = [item["input_ids"] for item in batch]
    prompt_lens = torch.tensor([item["prompt_len"] for item in batch])
    
    padded_input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    return {"input_ids": padded_input_ids, "prompt_lens": prompt_lens}

# ---- Main Training Loop -----------------------------------------------------

import json

def run_training():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Distributed setup (simplified)
    if "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        device = f"cuda:{dist.get_local_rank()}"
        torch.cuda.set_device(device)
    else:
        rank = 0
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = build_tokenizer(config["model"]["tokenizer_path"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        mask_token_id = 156895 # Fallback for LLaDA 2.0
    
    if rank == 0:
        print(f"Using Mask Token ID: {mask_token_id} ({tokenizer.decode([mask_token_id])})")

    dataset = SFTDataset(config["data"]["train_path"], tokenizer, config["train"]["max_seq_length"])
    dataloader = DataLoader(dataset, batch_size=config["train"]["per_device_batch_size"], 
                            collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
                            shuffle=True)

    model_path = config["model"]["model_path"]
    config_path = config["model"].get("config_path", model_path)
    if config_path is None: config_path = model_path

    model = build_foundation_model(
        weights_path=model_path,
        config_path=config_path,
        torch_dtype="bfloat16" if config["train"]["mixed_precision"] == "bf16" else "float32"
    )
    model.to(device)
    
    # Optional: build_parallelize_model(model, ...) for FSDP

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["train"]["learning_rate"]), 
                                 weight_decay=config["train"]["weight_decay"])
    
    num_training_steps = len(dataloader) * config["train"]["num_epochs"]
    lr_scheduler = get_scheduler(config["train"]["lr_scheduler"], optimizer, 
                                 num_warmup_steps=config["train"]["warmup_steps"], 
                                 num_training_steps=num_training_steps)

    model.train()
    for epoch in range(config["train"]["num_epochs"]):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", disable=rank != 0)
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            prompt_lengths = batch["prompt_lens"].to(device)

            # NEW — unit-level masking (Spec Section 6):
            masked_input_ids, labels = apply_response_unit_mask(input_ids, prompt_lengths, mask_token_id)
            
            # Bidirectional attention over all non-padding positions
            attn_mask = (masked_input_ids != tokenizer.pad_token_id).long()

            logits = model(input_ids=masked_input_ids, attention_mask=attn_mask).logits
            loss = compute_unit_mask_loss(logits, labels)

            loss.backward()
            
            if (pbar.n + 1) % config["train"].get("gradient_accumulation_steps", 1) == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["train"]["gradient_clip"])
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            pbar.set_postfix(loss=loss.item())

        # Save checkpoint
        if rank == 0:
            os.makedirs(config["train"]["output_dir"], exist_ok=True)
            model.save_pretrained(os.path.join(config["train"]["output_dir"], f"epoch_{epoch}"))

if __name__ == "__main__":
    run_training()
