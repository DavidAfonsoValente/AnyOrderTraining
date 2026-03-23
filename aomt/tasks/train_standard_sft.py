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

def apply_response_unit_mask(input_ids: torch.Tensor,
                              prompt_lengths: torch.Tensor,
                              mask_token_id: int) -> tuple:
    """LLaDA2/MDLM forward process: random masking ratio per sample.
    
    For each sample in the batch, samples t ~ U(0,1) and independently
    masks each response token with probability t. This ensures the model
    learns to denoise from all noise levels, matching the iterative
    block-wise generation process.
    """
    B, L = input_ids.shape
    positions = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
    response_mask = positions >= prompt_lengths.unsqueeze(1)

    # Sample random masking ratio t ~ U(0,1) per sample  
    t = torch.rand(B, 1, device=input_ids.device)
    # Independently mask each response token with probability t
    token_mask = (torch.rand(B, L, device=input_ids.device) < t) & response_mask

    masked_input_ids = input_ids.clone()
    masked_input_ids[token_mask] = mask_token_id

    labels = torch.full_like(input_ids, -100)
    labels[token_mask] = input_ids[token_mask]

    return masked_input_ids, labels

class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, jsonl_path, tokenizer, max_seq_length):
        with open(jsonl_path, "r") as f:
            self.examples = [json.loads(line) for line in f]
        self.tokenizer, self.max_seq_length = tokenizer, max_seq_length

    def __len__(self): return len(self.examples)
    def __getitem__(self, idx):
        ex = self.examples[idx]
        messages = ex["messages"]
        input_ids = self.tokenizer.apply_chat_template(messages, tokenize=True, truncation=True, max_length=self.max_seq_length)
        if not isinstance(input_ids, list):
            input_ids = input_ids["input_ids"] if hasattr(input_ids, "__getitem__") and "input_ids" in input_ids else list(input_ids)
        prompt_str = self.tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
        prompt_ids = self.tokenizer.encode(prompt_str, add_special_tokens=False)
        prompt_len = min(len(prompt_ids), len(input_ids))
        return {"input_ids": torch.tensor(input_ids, dtype=torch.long), "prompt_len": prompt_len}

def collate_fn(batch, pad_id):
    input_ids = torch.nn.utils.rnn.pad_sequence([item["input_ids"] for item in batch], batch_first=True, padding_value=pad_id)
    prompt_lens = torch.tensor([item["prompt_len"] for item in batch])
    return {"input_ids": input_ids, "prompt_lens": prompt_lens}

def main():
    if "RANK" in os.environ and not dist.is_initialized():
        dist.init_process_group(backend=get_dist_comm_backend())
    
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args_cli = parser.parse_args()

    with open(args_cli.config, "r") as f:
        config = yaml.safe_load(f)

    init_parallel_state(dp_size=dist.get_world_size() if dist.is_initialized() else 1,
                        dp_mode=config["train"].get("fsdp_type", "fsdp2"))
    ps = get_parallel_state()
    # Meta init only works with FSDP (world_size>1); single-GPU must load on CUDA.
    init_device = "meta" if ps.fsdp_enabled else "cuda"
    device_type = get_device_type()
    device_name = f"{device_type}:{ps.local_rank}"
    get_torch_device().set_device(device_name)
    
    tokenizer = build_tokenizer(config["model"]["tokenizer_path"])
    mask_token_id = tokenizer.mask_token_id or 156895
    
    dataset = SFTDataset(config["data"]["train_path"], tokenizer, config["train"]["max_seq_length"])
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
            input_ids, prompt_lengths = batch["input_ids"], batch["prompt_lens"]
            masked_input_ids, labels = apply_response_unit_mask(input_ids, prompt_lengths, mask_token_id)
            # Bidirectional 2D attention mask
            attn_mask = (masked_input_ids != (tokenizer.pad_token_id or 0)).to(torch.bfloat16 if use_bf16 else torch.float32)
            
            with torch.amp.autocast("cuda", enabled=use_bf16, dtype=torch.bfloat16):
                logits = model(input_ids=masked_input_ids, attention_mask=attn_mask, use_cache=False).logits
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
