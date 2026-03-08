"""
tasks/train_standard_sft.py
Baseline custom task for Standard SFT, Prefix SFT Stage 1, and Stage 2.
Refactored to align strictly with dFactory standards while implementing unit-level masking.
"""

import os
import torch
import torch.distributed as dist
import yaml
import argparse
from tqdm import trange
import numpy as np
import json
from functools import partial

# dFactory/VeOmni Imports
from veomni.models import build_foundation_model, build_tokenizer
from veomni.distributed.parallel_state import init_parallel_state, get_parallel_state
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.optim import build_lr_scheduler, build_optimizer
from veomni.utils import helper
from veomni.utils.device import get_device_type, get_nccl_backend, synchronize, get_torch_device
from veomni.utils.dist_utils import all_reduce
from veomni.models.registry import ModelRegistry

# Ensure custom modeling is registered
ModelRegistry.register_modeling_path("models.llada2_moe")

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

def main():
    dist.init_process_group(backend=get_nccl_backend())
    
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args_cli = parser.parse_args()

    with open(args_cli.config, "r") as f:
        config = yaml.safe_load(f)

    init_parallel_state(
        dp_size=dist.get_world_size(),
        dp_mode=config["train"].get("fsdp_type", "fsdp1")
    )
    
    ps = get_parallel_state()
    get_torch_device().set_device(f"{get_device_type()}:{ps.local_rank}")
    
    logger = helper.create_logger(__name__)
    
    tokenizer = build_tokenizer(config["model"]["tokenizer_path"])
    mask_token_id = tokenizer.mask_token_id or 156895
    
    dataset = SFTDataset(config["data"]["train_path"], tokenizer, config["train"]["max_seq_length"])
    
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=ps.world_size, rank=ps.global_rank, shuffle=True
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=config["train"]["per_device_batch_size"],
        sampler=sampler,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id or tokenizer.eos_token_id)
    )

    model = build_foundation_model(
        config_path=config["model"].get("config_path", config["model"]["model_path"]),
        weights_path=config["model"]["model_path"],
        torch_dtype="bfloat16" if config["train"]["mixed_precision"] == "bf16" else "float32",
        attn_implementation="sdpa",
        init_device=f"{get_device_type()}:{ps.local_rank}"
    )

    model = build_parallelize_model(
        model,
        weights_path=config["model"]["model_path"],
        enable_gradient_checkpointing=config["train"].get("gradient_checkpointing", True),
        enable_mixed_precision=(config["train"]["mixed_precision"] == "bf16"),
        basic_modules=["LLaDA2MoeDecoderLayer"]
    )

    optimizer = build_optimizer(
        model,
        lr=float(config["train"]["learning_rate"]),
        weight_decay=config["train"]["weight_decay"],
        fused=True
    )

    num_training_steps = len(dataloader) * config["train"]["num_epochs"]
    lr_scheduler = build_lr_scheduler(
        optimizer,
        train_steps=num_training_steps,
        lr=float(config["train"]["learning_rate"]),
        lr_min=float(config["train"].get("min_lr", 0)),
        lr_warmup_ratio=float(config["train"].get("warmup_steps", 0)) / num_training_steps if num_training_steps > 0 else 0
    )

    model.train()
    for epoch in range(config["train"]["num_epochs"]):
        sampler.set_epoch(epoch)
        pbar = trange(len(dataloader), desc=f"Epoch {epoch}", disable=ps.global_rank != 0)
        data_iter = iter(dataloader)
        
        for _ in pbar:
            batch = next(data_iter)
            batch = {k: v.to(get_torch_device().get_device_name(), non_blocking=True) for k, v in batch.items()}
            
            input_ids = batch["input_ids"]
            prompt_lengths = batch["prompt_lens"]

            # Standard SFT masking
            masked_input_ids, labels = apply_response_unit_mask(input_ids, prompt_lengths, mask_token_id)
            
            # Bidirectional 4D mask
            seq_len = masked_input_ids.shape[1]
            padding_mask = (masked_input_ids != (tokenizer.pad_token_id or tokenizer.eos_token_id)).long()
            attn_mask = padding_mask.unsqueeze(1).unsqueeze(2).expand(-1, 1, seq_len, seq_len)

            logits = model(input_ids=masked_input_ids, attention_mask=attn_mask, use_cache=False).logits
            
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )

            loss.backward()
            
            if hasattr(model, "clip_grad_norm_"):
                grad_norm = model.clip_grad_norm_(config["train"]["gradient_clip"])
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["train"]["gradient_clip"])

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            # Sync loss for logging
            reduced_loss = all_reduce(loss.detach(), group=ps.fsdp_group)
            
            if ps.global_rank == 0:
                pbar.set_postfix(loss=reduced_loss.item())

        if ps.global_rank == 0:
            save_path = os.path.join(config["train"]["output_dir"], f"epoch_{epoch}")
            os.makedirs(save_path, exist_ok=True)
            model.save_pretrained(save_path)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
