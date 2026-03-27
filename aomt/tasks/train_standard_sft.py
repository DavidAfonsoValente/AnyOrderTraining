"""
tasks/train_standard_sft.py
Baseline custom task for Standard SFT, Prefix SFT Stage 1, and Stage 2.
Refactored to align strictly with dFactory standards while implementing unit-level masking.
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
from unittest.mock import MagicMock

# Mock CUDA before any imports that might trigger CUDA checks
if not torch.cuda.is_available():
    torch.cuda.get_device_capability = MagicMock(return_value=(8, 0))
    torch.cuda.get_device_properties = MagicMock()
    torch.cuda.is_initialized = MagicMock(return_value=True)
    if not hasattr(torch._C, '_cuda_init'):
        torch._C._cuda_init = MagicMock()
    # Mock torch.cpu.get_device_name
    if not hasattr(torch.cpu, 'get_device_name'):
        torch.cpu.get_device_name = MagicMock(return_value="cpu")

os.environ.setdefault("MODELING_BACKEND", "hf")
os.environ["HF_HUB_TRUST_REMOTE_CODE"] = "1"
os.environ["TRUST_REMOTE_CODE"] = "1"

def compute_unit_mask_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Shared loss function: Cross-entropy over masked positions only."""
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
        reduction="mean",
    )

def apply_response_unit_mask(
    input_ids: torch.Tensor,       # [B, L] clean token ids from dFactory collator
    prompt_lengths: torch.Tensor,  # [B]    number of prompt tokens per example
    mask_token_id: int,
) -> tuple:
    """
    Deterministically mask ALL response tokens. Prompt tokens stay clean.
    """
    B, L = input_ids.shape
    device = input_ids.device
    pos = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)  # [B, L]
    is_response = pos >= prompt_lengths.unsqueeze(1)                  # [B, L]

    masked = input_ids.clone()
    masked[is_response] = mask_token_id

    labels = input_ids.clone()
    labels[~is_response] = -100

    return masked, labels

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
        if hasattr(input_ids, "input_ids"):
            input_ids = input_ids.input_ids
        prompt_ids = self.tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
        if hasattr(prompt_ids, "input_ids"):
            prompt_ids = prompt_ids.input_ids
        prompt_len = min(len(prompt_ids), len(input_ids))
        return {"input_ids": torch.tensor(input_ids, dtype=torch.long), "prompt_len": prompt_len}

def collate_fn(batch, pad_id):
    input_ids = torch.nn.utils.rnn.pad_sequence([item["input_ids"] for item in batch], batch_first=True, padding_value=pad_id)
    prompt_lens = torch.tensor([item["prompt_len"] for item in batch])
    return {"input_ids": input_ids, "prompt_lens": prompt_lens}

def main():
    # dFactory/VeOmni Imports
    from veomni.models import build_foundation_model, build_tokenizer, save_model_weights
    from veomni.distributed.parallel_state import init_parallel_state, get_parallel_state
    from veomni.distributed.torch_parallelize import build_parallelize_model
    from veomni.optim import build_lr_scheduler, build_optimizer
    from veomni.utils import helper
    from veomni.utils.device import get_device_type, get_dist_comm_backend, get_torch_device
    from veomni.utils.dist_utils import all_reduce

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args_cli = parser.parse_args()

    with open(args_cli.config, "r") as f:
        config = yaml.safe_load(f)

    if args_cli.device == "cpu":
        device_name = "cpu"
        ps_world_size = 1
        ps_global_rank = 0
        ps_fsdp_enabled = False
        init_device = "cpu"
        fsdp_group = None
    else:
        if "RANK" in os.environ and not dist.is_initialized():
            dist.init_process_group(backend=get_dist_comm_backend())
        init_parallel_state(dp_size=dist.get_world_size() if dist.is_initialized() else 1,
                            dp_mode=config["train"].get("fsdp_type", "fsdp2"))
        ps = get_parallel_state()
        ps_world_size = ps.world_size
        ps_global_rank = ps.global_rank
        ps_fsdp_enabled = ps.fsdp_enabled
        fsdp_group = ps.fsdp_group
        device_type = get_device_type()
        device_name = f"{device_type}:{ps.local_rank}"
        get_torch_device().set_device(device_name)
        init_device = "meta" if ps_fsdp_enabled else "cuda"
    
    tokenizer = build_tokenizer(config["model"]["tokenizer_path"])
    mask_token_id = tokenizer.mask_token_id or 156895
    
    dataset = SFTDataset(config["data"]["train_path"], tokenizer, config["train"]["max_seq_length"])
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=ps_world_size, rank=ps_global_rank, shuffle=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config["train"]["per_device_batch_size"], 
                                             sampler=sampler, collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id or 0))

    model = build_foundation_model(weights_path=config["model"]["model_path"],
                                   config_path=config["model"].get("config_path", config["model"]["model_path"]),
                                   torch_dtype="bfloat16" if config["train"]["mixed_precision"] == "bf16" else "float32",
                                   attn_implementation="sdpa", init_device=init_device,
                                   moe_implementation="fused")

    for param in model.parameters():
        param.requires_grad = True

    if not args_cli.device == "cpu":
        model = build_parallelize_model(model, weights_path=config["model"]["model_path"],
                                        enable_gradient_checkpointing=config["train"].get("gradient_checkpointing", True),
                                        enable_mixed_precision=False,
                                        basic_modules=["LLaDA2MoeDecoderLayer"],
                                        init_device=init_device)

    optimizer = build_optimizer(model, lr=float(config["train"]["learning_rate"]), weight_decay=config["train"]["weight_decay"], fused=(device_name != "cpu"))
    use_bf16 = (config["train"]["mixed_precision"] == "bf16") and (device_name != "cpu")
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
    global_step = 0
    for epoch in range(config["train"]["num_epochs"]):
        sampler.set_epoch(epoch)
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{config['train']['num_epochs']}", disable=(ps_global_rank != 0))
        for micro_step, batch in enumerate(pbar):
            batch = {k: v.to(device_name, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            input_ids, prompt_lengths = batch["input_ids"], batch["prompt_lens"]
            
            masked_input_ids, labels = apply_response_unit_mask(input_ids, prompt_lengths, mask_token_id)
            
            pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
            attn_mask = (input_ids != pad_id).long()
            
            autocast_device = "cuda" if "cuda" in device_name else "cpu"
            with torch.amp.autocast(autocast_device, enabled=use_bf16, dtype=torch.bfloat16):
                logits = model(input_ids=masked_input_ids, attention_mask=attn_mask, use_cache=False).logits
                loss = compute_unit_mask_loss(logits, labels)

            if global_step < 10 and ps_global_rank == 0:
                with torch.no_grad():
                    assert not loss.isnan(), f"NaN loss at step {global_step}"
                    assert loss.item() > 0, f"Zero loss at step {global_step} — check labels"
                    b0_plen = prompt_lengths[0].item()
                    assert (masked_input_ids[0, :b0_plen] == input_ids[0, :b0_plen]).all(), "BUG: prompt tokens masked"
                    if b0_plen < masked_input_ids.shape[1]:
                        assert (masked_input_ids[0, b0_plen:] == mask_token_id).all(), "BUG: response not fully masked"

            scaled_loss = loss / accum_steps
            scaler.scale(scaled_loss).backward()

            if (micro_step + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["train"]["gradient_clip"])
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            if fsdp_group:
                avg_loss = all_reduce(loss.detach(), group=fsdp_group)
            else:
                avg_loss = loss.item()
            
            if ps_global_rank == 0: pbar.set_postfix(loss=f"{avg_loss:.4f}")
            
            if args_cli.max_steps and global_step >= args_cli.max_steps:
                break
        if args_cli.max_steps and global_step >= args_cli.max_steps:
            break

        if ps_global_rank == 0:
            save_path = os.path.join(config["train"]["output_dir"], f"epoch_{epoch}")
            save_model_weights(save_path, model.state_dict(), global_rank=ps_global_rank)

    if dist.is_initialized(): dist.destroy_process_group()

if __name__ == "__main__": main()
