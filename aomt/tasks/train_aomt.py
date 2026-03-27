"""
tasks/train_aomt.py
AOMT Mixed training for agent trajectories.
Strictly follows dFactory standards and Engineering Spec v3.
"""

import os
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
import argparse
from tqdm import tqdm
import numpy as np
import json

# aomt imports
from aomt.utils.tokenizer_utils import get_mask_token_id

os.environ.setdefault("MODELING_BACKEND", "hf")
os.environ["HF_HUB_TRUST_REMOTE_CODE"] = "1"
os.environ["TRUST_REMOTE_CODE"] = "1"

def compute_unit_mask_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over labeled (masked) positions only."""
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
        reduction="mean",
    )

def apply_unit_mask(
    unit_texts: list,    # list of strings, alternating obs/act
    unit_types: list,    # list of "obs" or "act"
    tokenizer,
    mask_prob: float,    # 0.25 default
    rng,                 # numpy Generator — must be freshly seeded each call
    sep_token_id: int,   # tokenizer.eos_token_id — unit separator
    mask_token_id: int,  # 156895
) -> tuple:
    """
    Tokenise the full trajectory flat with EOS separators between units.
    Matches Spec: O_0 [EOS] A_0 [EOS] ... O_T [EOS]
    Apply unit-level Bernoulli(mask_prob) masking to ALL units.
    """
    all_ids: list = []
    spans:   list = []   # (start_inclusive, end_exclusive, unit_type)

    for i, (text, utype) in enumerate(zip(unit_texts, unit_types)):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            continue
        start = len(all_ids)
        all_ids.extend(ids)
        end = len(all_ids)
        spans.append((start, end, utype))
        # EOS separator AFTER every unit including the last
        all_ids.append(sep_token_id)

    if not all_ids:
        dummy = torch.zeros(1, dtype=torch.long)
        return dummy, torch.full_like(dummy, -100)

    input_ids = torch.tensor(all_ids, dtype=torch.long)
    labels    = torch.full_like(input_ids, -100)

    # AOMT-Mixed: all units are eligible for masking
    eligible = list(spans)

    if not eligible:
        return input_ids, labels

    # Bernoulli masking
    masked_any = False
    for start, end, _ in eligible:
        if rng.random() < mask_prob:
            labels[start:end] = input_ids[start:end].clone()
            input_ids[start:end] = mask_token_id
            masked_any = True

    # Force-mask fallback to avoid zero-loss steps
    if not masked_any:
        idx = int(rng.integers(len(eligible)))
        s, e, _ = eligible[idx]
        labels[s:e] = input_ids[s:e].clone()
        input_ids[s:e] = mask_token_id

    return input_ids, labels

class AOMTDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        jsonl_path: str,
        tokenizer,
        mask_prob: float,      # 0.25
        mask_token_id: int,    # 156895
        max_seq_length: int = 2048,
    ):
        self.tokenizer      = tokenizer
        self.mask_prob      = mask_prob
        self.mask_token_id  = mask_token_id
        self.max_seq_length = max_seq_length
        self.sep_id         = tokenizer.eos_token_id

        self.examples = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                if "unit_texts" not in d or "unit_types" not in d:
                    continue
                # Ensure at least one action turn
                n_act = sum(1 for t in d["unit_types"] if t == "act")
                if n_act == 0:
                    continue
                self.examples.append(d)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex  = self.examples[idx]
        rng = np.random.default_rng()   # true OS entropy — different every call

        input_ids, labels = apply_unit_mask(
            unit_texts    = ex["unit_texts"],
            unit_types    = ex["unit_types"],
            tokenizer     = self.tokenizer,
            mask_prob     = self.mask_prob,
            rng           = rng,
            sep_token_id  = self.sep_id,
            mask_token_id = self.mask_token_id,
        )

        if len(input_ids) > self.max_seq_length:
            input_ids = input_ids[:self.max_seq_length]
            labels    = labels[:self.max_seq_length]

        return {"input_ids": input_ids, "labels": labels}

def collate_fn(batch, pad_id):
    input_ids = torch.nn.utils.rnn.pad_sequence([item["input_ids"] for item in batch], batch_first=True, padding_value=pad_id)
    labels = torch.nn.utils.rnn.pad_sequence([item["labels"] for item in batch], batch_first=True, padding_value=-100)
    return {"input_ids": input_ids, "labels": labels}

def main():
    # dFactory/VeOmni Imports
    from veomni.models import build_foundation_model, build_tokenizer, save_model_weights
    from veomni.distributed.parallel_state import init_parallel_state, get_parallel_state
    from veomni.distributed.torch_parallelize import build_parallelize_model
    from veomni.optim import build_lr_scheduler, build_optimizer
    from veomni.utils import helper
    from veomni.utils.device import get_device_type, get_dist_comm_backend, get_torch_device
    from veomni.utils.dist_utils import all_reduce

    if "RANK" in os.environ and not dist.is_initialized():
        dist.init_process_group(backend=get_dist_comm_backend())
    
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    # Expose mask_prob for ablation sweep
    parser.add_argument("--mask_prob", type=float, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # CLI override for mask_prob
    mask_prob = args.mask_prob if args.mask_prob is not None else config["aomt"].get("mask_prob", 0.25)

    init_parallel_state(dp_size=dist.get_world_size() if dist.is_initialized() else 1,
                        dp_mode=config["train"].get("fsdp_type", "fsdp2"))
    ps = get_parallel_state()
    init_device = "meta" if ps.fsdp_enabled else "cuda"
    device_type = get_device_type()
    device_name = f"{device_type}:{ps.local_rank}"
    get_torch_device().set_device(device_name)
    
    tokenizer = build_tokenizer(config["model"]["tokenizer_path"])
    mask_token_id = get_mask_token_id(config["model"]["tokenizer_path"])
    
    dataset = AOMTDataset(
        config["data"]["train_path"], 
        tokenizer, 
        mask_prob, 
        mask_token_id,
        max_seq_length=config["train"].get("max_seq_length", 2048)
    )
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=ps.world_size, rank=ps.global_rank, shuffle=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config["train"]["per_device_batch_size"], 
                                             sampler=sampler, collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id or 0))

    model = build_foundation_model(weights_path=config["model"]["model_path"],
                                   config_path=config["model"].get("config_path", config["model"]["model_path"]),
                                   torch_dtype="bfloat16" if config["train"]["mixed_precision"] == "bf16" else "float32",
                                   attn_implementation="sdpa", init_device=init_device,
                                   moe_implementation="fused")

    for param in model.parameters():
        param.requires_grad = True

    model = build_parallelize_model(model, weights_path=config["model"]["model_path"],
                                    enable_gradient_checkpointing=config["train"].get("gradient_checkpointing", True),
                                    enable_mixed_precision=False,
                                    basic_modules=["LLaDA2MoeDecoderLayer"],
                                    init_device=init_device)

    optimizer = build_optimizer(model, lr=float(config["train"]["learning_rate"]), weight_decay=config["train"]["weight_decay"], fused=True)
    use_bf16 = (config["train"]["mixed_precision"] == "bf16")
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    accum_steps = config["train"].get("gradient_accumulation_steps", 1)
    micro_batches_per_epoch = len(dataloader)
    num_steps = (micro_batches_per_epoch * config["train"]["num_epochs"]) // accum_steps
    warmup_steps = int(config["train"].get("warmup_steps", 0))
    warmup_ratio = warmup_steps / num_steps if num_steps > 0 else 0
    scheduler = build_lr_scheduler(optimizer, train_steps=num_steps, lr=float(config["train"]["learning_rate"]),
                                   lr_min=float(config["train"].get("min_lr", 0)),
                                   lr_warmup_ratio=warmup_ratio)

    model.train()
    for epoch in range(config["train"]["num_epochs"]):
        sampler.set_epoch(epoch)
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{config['train']['num_epochs']}", disable=(ps.global_rank != 0))
        for micro_step, batch in enumerate(pbar):
            batch = {k: v.to(device_name, non_blocking=True) for k, v in batch.items()}
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            
            pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
            attn_mask = (input_ids != pad_id).long()
            
            with torch.amp.autocast("cuda", enabled=use_bf16, dtype=torch.bfloat16):
                logits = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False).logits
                loss = compute_unit_mask_loss(logits, labels)

            scaled_loss = loss / accum_steps
            scaler.scale(scaled_loss).backward()

            if (micro_step + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["train"]["gradient_clip"])
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            avg_loss = all_reduce(loss.detach(), group=ps.fsdp_group)
            if ps.global_rank == 0: pbar.set_postfix(loss=f"{avg_loss:.4f}")

        save_path = os.path.join(config["train"]["output_dir"], f"epoch_{epoch}")
        save_model_weights(save_path, model.state_dict(), global_rank=ps.global_rank)

    if dist.is_initialized(): dist.destroy_process_group()

if __name__ == "__main__": main()

