# aomt/training/trainer.py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from tqdm import tqdm
import yaml
import os
import argparse
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
import functools

from aomt.data.unit_parser import TokenizedTrajectory
from aomt.training.mask_sampler import MaskMode, apply_unit_mask
from aomt.training.objectives import masked_unit_cross_entropy
from aomt.training.collator import AOMTDataCollator, build_prefix_sft_examples

# --- Section 7.1: Dataset Class ---

class AOMTDataset(Dataset):
    """
    A PyTorch Dataset for Any-Order Masked Training.
    """
    def __init__(
        self,
        tokenized_trajectories: list[TokenizedTrajectory],
        tokenizer: AutoTokenizer,
        mode: MaskMode,
        mask_prob: float,
        seed: int = 42,
    ):
        self.trajectories = tokenized_trajectories
        self.tokenizer = tokenizer
        self.mode = mode
        self.mask_prob = mask_prob
        self.mask_token_id = tokenizer.mask_token_id
        self.base_rng = np.random.default_rng(seed)

        if self.mask_token_id is None:
            raise ValueError("Tokenizer must have a `mask_token_id`.")

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> dict:
        traj = self.trajectories[idx]
        rng = np.random.default_rng(self.base_rng.integers(1e9))
        masked_ids, loss_mask = apply_unit_mask(
            traj, self.mask_prob, self.mode, self.mask_token_id, rng
        )
        return {
            "input_ids": masked_ids,
            "target_ids": traj.input_ids.clone(),
            "loss_mask": loss_mask,
            "use_causal_mask": (self.mode == MaskMode.STANDARD_SFT),
        }

class PrefixSFTDataset(Dataset):
    """
    A specialized PyTorch Dataset for the Prefix-SFT (ALEE-style) Stage 1.
    """
    def __init__(
        self,
        tokenized_trajectories: list[TokenizedTrajectory],
        tokenizer: AutoTokenizer
    ):
        self.examples = []
        for traj in tqdm(tokenized_trajectories):
            self.examples.extend(build_prefix_sft_examples(traj, tokenizer))
        print(f"Created {len(self.examples)} Prefix-SFT examples.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]

# --- Section 7.2: Training Step ---

def training_step(model, batch, device):
    """
    Performs a single training step (forward pass, loss calculation).
    """
    for key, val in batch.items():
        if isinstance(val, torch.Tensor):
            batch[key] = val.to(device)

    is_causal = batch["use_causal_mask"][0].item()
    
    # Let the model itself handle attention mask creation
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
    )
    
    loss = masked_unit_cross_entropy(outputs.logits, batch["target_ids"], batch["loss_mask"])
    return loss

# --- Main Training Loop (dFactory placeholder) ---

def run_training(config_path: str, is_distributed: bool):
    """
    Main training function, updated to handle FSDP.
    """
    # 1. Distributed Setup
    rank = 0
    world_size = 1
    if is_distributed:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(rank)

    if rank == 0:
        print(f"--- Training Config: {config_path} ---")
        print(f"Distributed: {is_distributed}, World Size: {world_size}")
        with open(config_path, 'r') as f:
            print(yaml.safe_load(f))
        print("--------------------")

    # 2. Load Config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 3. Setup Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model"])
    if tokenizer.pad_token is None: tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    if tokenizer.mask_token is None: tokenizer.add_special_tokens({'mask_token': '[MASK]'})

    # 4. Load Data
    if rank == 0:
        if not os.path.exists(config.get("data_cache_path")):
             raise FileNotFoundError(f"Cache not found at {config.get('data_cache_path')}. Run parse_trajectories.py first.")
    
    tokenized_trajectories = torch.load(config.get("data_cache_path"))

    # 5. Initialize Dataset, Sampler, DataLoader
    mask_mode = MaskMode(config["mask_mode"])
    dataset = AOMTDataset(tokenized_trajectories, tokenizer, mode=mask_mode, mask_prob=config.get("mask_prob", 0.0))
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if is_distributed else None
    collator = AOMTDataCollator(tokenizer)
    dataloader = DataLoader(dataset, batch_size=config.get("per_device_batch_size", 2), collate_fn=collator, sampler=sampler, shuffle=(sampler is None))

    # 6. Initialize Model
    # FSDP requires a specific wrapping policy.
    auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=1_000_000
    )
    
    model = AutoModelForCausalLM.from_pretrained(config["model"], torch_dtype=torch.bfloat16)
    model.resize_token_embeddings(len(tokenizer))

    if is_distributed:
        model = FSDP(model, auto_wrap_policy=auto_wrap_policy, device_id=rank)
    else:
        model = model.to(rank)

    # 7. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    lr_scheduler = get_scheduler(name=config.get("lr_schedule", "cosine"), optimizer=optimizer, num_warmup_steps=config.get("warmup_steps", 0), num_training_steps=len(dataloader) * config["num_epochs"])

    # 8. Training Loop
    if rank == 0: print("Starting training...")
    for epoch in range(config["num_epochs"]):
        if rank == 0: print(f"\n--- Epoch {epoch+1}/{config['num_epochs']} ---")
        if is_distributed: sampler.set_epoch(epoch)
        
        model.train()
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}", disable=(rank != 0))

        for batch in progress_bar:
            loss = training_step(model, batch, rank)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            if rank == 0: progress_bar.set_postfix(loss=loss.item())

    if rank == 0: print("\n--- Training Complete ---")
    if is_distributed: dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the experiment config file.")
    parser.add_argument("--distributed", action='store_true', help="Flag to enable distributed training.")
    args = parser.parse_args()
    
    run_training(args.config, args.distributed)
