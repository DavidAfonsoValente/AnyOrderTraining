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
from veomni.utils.device import get_device_type, get_dist_comm_backend, get_torch_device
from veomni.utils.dist_utils import all_reduce

os.environ.setdefault("MODELING_BACKEND", "hf")
os.environ["HF_HUB_TRUST_REMOTE_CODE"] = "1"
os.environ["TRUST_REMOTE_CODE"] = "1"

# aomt imports
from aomt.data.unit_parser import parse_conversation_to_trajectory, tokenize_trajectory
from aomt.training.mask_sampler import apply_unit_mask, MaskMode

def compute_unit_mask_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Shared loss function: Cross-entropy over masked positions only."""
    mask = (labels != -100)
    if not mask.any():
        return logits.flatten()[0] * 0.0
    return torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100
    )

class AOMTDataset(torch.utils.data.Dataset):
    def __init__(self, jsonl_path, tokenizer, mask_prob, mode, mask_token_id, seed=42):
        with open(jsonl_path, "r") as f:
            self.examples = [json.loads(line) for line in f]
        self.tokenizer = tokenizer
        self.mask_prob = mask_prob
        self.mode = MaskMode(mode)
        self.mask_token_id = mask_token_id
        self.seed = seed

    def __len__(self): return len(self.examples)
    def __getitem__(self, idx):
        ex = self.examples[idx]
        # Reconstruct the trajectory object
        traj = parse_conversation_to_trajectory(ex)
        # Tokenize with chat template
        tokenized_traj = tokenize_trajectory(traj, self.tokenizer)
        
        if tokenized_traj is None:
            # Fallback for very long trajectories that truncation failed on
            return self.__getitem__((idx + 1) % len(self.examples))

        rng = np.random.default_rng([self.seed, idx, np.random.randint(0, 2**31)])
        input_ids, labels_mask = apply_unit_mask(tokenized_traj, self.mask_prob, self.mode, self.mask_token_id, rng)
        
        # Convert boolean loss_mask to labels (-100 for unmasked)
        labels = torch.full_like(input_ids, -100)
        labels[labels_mask] = tokenized_traj.input_ids[labels_mask]
        
        return {"input_ids": input_ids, "labels": labels}

def collate_fn(batch, pad_id):
    input_ids = torch.nn.utils.rnn.pad_sequence([item["input_ids"] for item in batch], batch_first=True, padding_value=pad_id)
    labels = torch.nn.utils.rnn.pad_sequence([item["labels"] for item in batch], batch_first=True, padding_value=-100)
    return {"input_ids": input_ids, "labels": labels}

def main():
    if "RANK" in os.environ and not dist.is_initialized():
        dist.init_process_group(backend=get_dist_comm_backend())
    
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    init_parallel_state(dp_size=dist.get_world_size() if dist.is_initialized() else 1,
                        dp_mode=config["train"].get("fsdp_type", "fsdp2"))
    ps = get_parallel_state()
    # VeOmni: fsdp_enabled only when world_size>1. Meta init requires FSDP; single-GPU must load on CUDA.
    init_device = "meta" if ps.fsdp_enabled else "cuda"
    device_type = get_device_type()
    device_name = f"{device_type}:{ps.local_rank}"
    get_torch_device().set_device(device_name)
    
    tokenizer = build_tokenizer(config["model"]["tokenizer_path"])
    mask_token_id = tokenizer.mask_token_id or 156895
    
    dataset = AOMTDataset(config["data"]["train_path"], tokenizer, config["aomt"]["mask_prob"], config["aomt"]["mode"], mask_token_id)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=ps.world_size, rank=ps.global_rank, shuffle=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config["train"]["per_device_batch_size"], 
                                             sampler=sampler, collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id or 0))

    # Memory optimization for single-GPU or fragmented environments
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    torch.cuda.empty_cache()

    model = build_foundation_model(weights_path=config["model"]["model_path"],
                                   config_path=config["model"].get("config_path", config["model"]["model_path"]),
                                   torch_dtype="bfloat16" if config["train"]["mixed_precision"] == "bf16" else "float32",
                                   attn_implementation="sdpa", init_device=init_device,
                                   moe_implementation="fused")

    # Ensure parameters require grad BEFORE parallelization
    for param in model.parameters():
        param.requires_grad = True

    model = build_parallelize_model(model, weights_path=config["model"]["model_path"],
                                    enable_gradient_checkpointing=config["train"].get("gradient_checkpointing", True),
                                    enable_mixed_precision=False,
                                    basic_modules=["LLaDA2MoeDecoderLayer"],
                                    init_device=init_device)

    # Diagnostic: Check precision of a few parameters
    if ps.global_rank == 0:
        for name, param in list(model.named_parameters())[:5]:
            print(f"[RE-CHECK] Param {name} dtype: {param.dtype}, requires_grad: {param.requires_grad}")

    optimizer = build_optimizer(model, lr=float(config["train"]["learning_rate"]), weight_decay=config["train"]["weight_decay"], fused=True)
    
    # Mixed precision setup: bf16 does NOT need a scaler
    use_bf16 = (config["train"]["mixed_precision"] == "bf16")
    # For bf16, disable the scaler entirely to avoid 'does not require grad' issues
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    from tqdm import tqdm
    accum_steps = config["train"].get("gradient_accumulation_steps", 1)

    # num_steps = total optimizer steps (accounting for gradient accumulation)
    # Paper: "50 warmup steps" — must compute ratio against optimizer steps
    micro_batches_per_epoch = len(dataloader)
    num_steps = (micro_batches_per_epoch * config["train"]["num_epochs"]) // accum_steps
    warmup_steps = int(config["train"].get("warmup_steps", 0))
    warmup_ratio = warmup_steps / num_steps if num_steps > 0 else 0
    scheduler = build_lr_scheduler(optimizer, train_steps=num_steps, lr=float(config["train"]["learning_rate"]),
                                   lr_min=float(config["train"].get("min_lr", 0)),
                                   lr_warmup_ratio=warmup_ratio)
    if ps.global_rank == 0:
        print(f"Gradient accumulation steps: {accum_steps}")
        print(f"Effective batch size: {config['train']['per_device_batch_size']} × {ps.world_size} × {accum_steps} = {config['train']['per_device_batch_size'] * ps.world_size * accum_steps}")

    model.train()
    for epoch in range(config["train"]["num_epochs"]):
        sampler.set_epoch(epoch)
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{config['train']['num_epochs']}", disable=(ps.global_rank != 0))
        for micro_step, batch in enumerate(pbar):
            batch = {k: v.to(device_name, non_blocking=True) for k, v in batch.items()}
            labels = batch.pop("labels")
            # Bidirectional 2D attention mask
            batch["attention_mask"] = (batch["input_ids"] != (tokenizer.pad_token_id or 0)).to(torch.bfloat16 if use_bf16 else torch.float32)
            
            with torch.amp.autocast("cuda", enabled=use_bf16, dtype=torch.bfloat16):
                logits = model(**batch, use_cache=False).logits
                loss = compute_unit_mask_loss(logits, labels)

            # Scale loss by accumulation steps so the gradient magnitude
            # is correct when accumulated over multiple micro-batches
            scaled_loss = loss / accum_steps
            scaler.scale(scaled_loss).backward()

            # Only step the optimizer every accum_steps micro-batches
            if (micro_step + 1) % accum_steps == 0:
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

            avg_loss = all_reduce(loss.detach(), group=ps.fsdp_group)
            if ps.global_rank == 0: pbar.set_postfix(loss=f"{avg_loss:.4f}")

        save_path = os.path.join(config["train"]["output_dir"], f"epoch_{epoch}")
        save_model_weights(save_path, model.state_dict(), global_rank=ps.global_rank)

    if dist.is_initialized(): dist.destroy_process_group()

if __name__ == "__main__": main()
