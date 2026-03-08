"""
tasks/train_aomt.py
AOMT custom task (Action-Only, Mixed).
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
from veomni.models import build_foundation_model, build_tokenizer, save_model_assets, save_model_weights
from veomni.distributed.parallel_state import init_parallel_state, get_parallel_state
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.optim import build_lr_scheduler, build_optimizer
from veomni.utils import helper
from veomni.utils.device import get_device_type, get_nccl_backend, synchronize, get_torch_device
from veomni.utils.dist_utils import all_reduce
from veomni.checkpoint import build_checkpointer
from veomni.models.registry import ModelRegistry

# Ensure custom modeling is registered
ModelRegistry.register_modeling_path("models.llada2_moe")

# ---- Spec Section 7.2: Masking logic ------------------------------------------

def apply_unit_mask(unit_texts: list,
                    unit_types: list,
                    tokenizer,
                    mask_prob: float,
                    mode: str,
                    rng,
                    mask_token_id: int) -> tuple:
    """
    Tokenises using the chat template and applies unit-level Bernoulli masking.
    """
    messages = []
    for text, utype in zip(unit_texts, unit_types):
        role = "user" if utype == "obs" else "assistant"
        messages.append({"role": role, "content": text})

    all_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    input_ids = torch.tensor(all_ids, dtype=torch.long)
    labels = torch.full_like(input_ids, -100)

    spans = []
    for i in range(1, len(messages) + 1):
        prefix_ids = tokenizer.apply_chat_template(messages[:i], tokenize=True, add_generation_prompt=False)
        start = spans[-1][1] if spans else 0
        end = len(prefix_ids)
        if end > len(all_ids): end = len(all_ids)
        utype = unit_types[i-1]
        spans.append((start, end, utype))

    masked_any = False
    for start, end, utype in spans:
        if start >= end: continue
        if mode == "action_only" and utype != "act":
            continue
        if rng.random() < mask_prob:
            labels[start:end] = input_ids[start:end].clone()
            input_ids[start:end] = mask_token_id
            masked_any = True

    if not masked_any:
        eligible = [(s, e) for s, e, ut in spans if (mode == "mixed" or ut == "act") and s < e]
        if eligible:
            s, e = eligible[rng.integers(len(eligible))]
            labels[s:e] = input_ids[s:e].clone()
            input_ids[s:e] = mask_token_id

    return input_ids, labels

def compute_unit_mask_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Shared loss function: Cross-entropy over masked positions only.
    Handles the case where all labels are -100 to avoid NaN.
    """
    if (labels == -100).all():
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    return torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100
    )

# ---- Spec Section 7.3: AOMT Dataset class ------------------------------------

class AOMTDataset(torch.utils.data.Dataset):
    def __init__(self, jsonl_path, tokenizer, mask_prob, mode, mask_token_id, seed=42):
        self.examples = []
        with open(jsonl_path, "r") as f:
            for line in f:
                self.examples.append(json.loads(line))
        self.tokenizer = tokenizer
        self.mask_prob = mask_prob
        self.mode = mode
        self.mask_token_id = mask_token_id
        self.seed = seed

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        rng = np.random.default_rng([self.seed, idx, np.random.randint(0, 2**31)])
        input_ids, labels = apply_unit_mask(
            ex["unit_texts"], ex["unit_types"],
            self.tokenizer, self.mask_prob, self.mode, rng,
            self.mask_token_id
        )
        return {"input_ids": input_ids, "labels": labels}

def collate_fn(batch, pad_token_id):
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    padded_input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    padded_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    return {"input_ids": padded_input_ids, "labels": padded_labels}

# ---- Main Training Loop -----------------------------------------------------

def main():
    dist.init_process_group(backend=get_nccl_backend())
    
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args_cli = parser.parse_args()

    with open(args_cli.config, "r") as f:
        config = yaml.safe_load(f)

    # Align config with dFactory expected structure for internal builders
    # (Assuming config matches spec v3 which uses dFactory-compatible fields)
    
    init_parallel_state(
        dp_size=dist.get_world_size(), # Simplified for now
        dp_mode=config["train"].get("fsdp_type", "fsdp1")
    )
    
    ps = get_parallel_state()
    get_torch_device().set_device(f"{get_device_type()}:{ps.local_rank}")
    
    logger = helper.create_logger(__name__)
    
    tokenizer = build_tokenizer(config["model"]["tokenizer_path"])
    mask_token_id = tokenizer.mask_token_id or 156895
    
    dataset = AOMTDataset(config["data"]["train_path"], tokenizer, 
                          config["aomt"]["mask_prob"], config["aomt"]["mode"],
                          mask_token_id)
    
    # Use dFactory dataloader if possible, or standard with ps info
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
            labels = batch.pop("labels")
            
            # Bidirectional 4D mask
            seq_len = batch["input_ids"].shape[1]
            padding_mask = (batch["input_ids"] != (tokenizer.pad_token_id or tokenizer.eos_token_id)).long()
            batch["attention_mask"] = padding_mask.unsqueeze(1).unsqueeze(2).expand(-1, 1, seq_len, seq_len)

            logits = model(**batch, use_cache=False).logits
            
            loss = compute_unit_mask_loss(logits, labels)

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
            # In a real run, use Checkpointer for FSDP state dicts
            model.save_pretrained(save_path)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
