"""
tasks/train_aomt.py
AOMT custom task (Action-Only, Mixed).
Strictly follows dFactory standards and Engineering Spec v3.
"""

import os
import torch
import torch.distributed as dist
import yaml
import argparse
from tqdm import trange
import numpy as np
import json

# dFactory/VeOmni Imports
from veomni.models import build_foundation_model, build_tokenizer, save_model_weights
from veomni.distributed.parallel_state import init_parallel_state, get_parallel_state
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.optim import build_lr_scheduler, build_optimizer
from veomni.utils import helper
from veomni.utils.device import get_device_type, get_nccl_backend, get_torch_device
from veomni.utils.dist_utils import all_reduce
from veomni.models.registry import ModelRegistry

# Register custom architecture
ModelRegistry.register_modeling_path("models.llada2_moe")

def compute_unit_mask_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Shared loss function: Cross-entropy over masked positions only."""
    mask = (labels != -100)
    if not mask.any():
        return logits.new_zeros(())
    return torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100
    )

def apply_unit_mask(unit_texts: list,
                    unit_types: list,
                    tokenizer,
                    mask_prob: float,
                    mode: str,
                    rng,
                    mask_token_id: int) -> tuple:
    """Unit-level Bernoulli masking using chat template boundaries."""
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
        spans.append((start, end, unit_types[i-1]))

    masked_any = False
    for start, end, utype in spans:
        if start >= end: continue
        if mode == "action_only" and utype != "act": continue
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

class AOMTDataset(torch.utils.data.Dataset):
    def __init__(self, jsonl_path, tokenizer, mask_prob, mode, mask_token_id, seed=42):
        with open(jsonl_path, "r") as f:
            self.examples = [json.loads(line) for line in f]
        self.tokenizer = tokenizer
        self.mask_prob, self.mode, self.mask_token_id = mask_prob, mode, mask_token_id
        self.seed = seed

    def __len__(self): return len(self.examples)
    def __getitem__(self, idx):
        ex = self.examples[idx]
        rng = np.random.default_rng([self.seed, idx, np.random.randint(0, 2**31)])
        input_ids, labels = apply_unit_mask(ex["unit_texts"], ex["unit_types"], self.tokenizer, 
                                            self.mask_prob, self.mode, rng, self.mask_token_id)
        return {"input_ids": input_ids, "labels": labels}

def collate_fn(batch, pad_id):
    input_ids = torch.nn.utils.rnn.pad_sequence([item["input_ids"] for item in batch], batch_first=True, padding_value=pad_id)
    labels = torch.nn.utils.rnn.pad_sequence([item["labels"] for item in batch], batch_first=True, padding_value=-100)
    return {"input_ids": input_ids, "labels": labels}

def main():
    if "RANK" in os.environ and not dist.is_initialized():
        dist.init_process_group(backend=get_nccl_backend())
    
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    init_parallel_state(dp_size=dist.get_world_size() if dist.is_initialized() else 1,
                        dp_mode=config["train"].get("fsdp_type", "fsdp1"))
    ps = get_parallel_state()
    device_type = get_device_type()
    device_name = f"{device_type}:{ps.local_rank}"
    get_torch_device().set_device(device_name)
    
    tokenizer = build_tokenizer(config["model"]["tokenizer_path"])
    mask_token_id = tokenizer.mask_token_id or 156895
    
    dataset = AOMTDataset(config["data"]["train_path"], tokenizer, config["aomt"]["mask_prob"], config["aomt"]["mode"], mask_token_id)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=ps.world_size, rank=ps.global_rank, shuffle=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config["train"]["per_device_batch_size"], 
                                             sampler=sampler, collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id or 0))

    model = build_foundation_model(weights_path=config["model"]["model_path"],
                                   config_path=config["model"].get("config_path", config["model"]["model_path"]),
                                   torch_dtype="bfloat16" if config["train"]["mixed_precision"] == "bf16" else "float32",
                                   attn_implementation="sdpa", init_device=device_type)

    model = build_parallelize_model(model, weights_path=config["model"]["model_path"],
                                    enable_gradient_checkpointing=config["train"].get("gradient_checkpointing", True),
                                    enable_mixed_precision=(config["train"]["mixed_precision"] == "bf16"),
                                    basic_modules=["LLaDA2MoeDecoderLayer"],
                                    init_device=device_type)

    optimizer = build_optimizer(model, lr=float(config["train"]["learning_rate"]), weight_decay=config["train"]["weight_decay"], fused=True)
    
    # Mixed precision setup
    use_bf16 = (config["train"]["mixed_precision"] == "bf16")
    scaler = torch.amp.GradScaler("cuda", enabled=(not use_bf16 and config["train"]["mixed_precision"] != "fp32"))

    num_steps = len(dataloader) * config["train"]["num_epochs"]
    scheduler = build_lr_scheduler(optimizer, train_steps=num_steps, lr=float(config["train"]["learning_rate"]),
                                   lr_min=float(config["train"].get("min_lr", 0)),
                                   lr_warmup_ratio=float(config["train"].get("warmup_steps", 0))/num_steps if num_steps > 0 else 0)

    model.train()
    for epoch in range(config["train"]["num_epochs"]):
        sampler.set_epoch(epoch)
        for batch in dataloader:
            batch = {k: v.to(device_name, non_blocking=True) for k, v in batch.items()}
            labels = batch.pop("labels")
            # Bidirectional 2D attention mask
            batch["attention_mask"] = (batch["input_ids"] != (tokenizer.pad_token_id or 0)).to(torch.bfloat16 if use_bf16 else torch.float32)
            
            with torch.amp.autocast("cuda", enabled=use_bf16, dtype=torch.bfloat16):
                logits = model(**batch, use_cache=False).logits
                loss = compute_unit_mask_loss(logits, labels)

            scaler.scale(loss).backward()
            
            if hasattr(model, "clip_grad_norm_"): 
                scaler.unscale_(optimizer)
                model.clip_grad_norm_(config["train"]["gradient_clip"])
            else: 
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["train"]["gradient_clip"])

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            if ps.global_rank == 0: print(f"Epoch {epoch} Loss: {all_reduce(loss.detach(), group=ps.fsdp_group).item():.4f}", end="\r")

        if ps.global_rank == 0:
            save_path = os.path.join(config["train"]["output_dir"], f"epoch_{epoch}")
            save_model_weights(model, save_path, model.state_dict())

    if dist.is_initialized(): dist.destroy_process_group()

if __name__ == "__main__": main()
