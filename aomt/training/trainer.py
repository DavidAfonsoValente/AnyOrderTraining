# aomt/training/trainer.py
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from tqdm import tqdm
import yaml
import argparse
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
import functools
import itertools
import wandb

from aomt.data.unit_parser import TokenizedTrajectory, TokenizedUnit
from aomt.training.mask_sampler import MaskMode, apply_unit_mask
from aomt.training.objectives import masked_unit_cross_entropy
from aomt.training.collator import AOMTDataCollator, build_prefix_sft_examples, build_prefix_sft_stage2_examples, build_standard_sft_examples
from datasets import load_from_disk
from collections import defaultdict
from torch.utils.data import Sampler

# --- Section 7.1: Dataset Class ---

class AOMTDataset(Dataset):
    """
    A PyTorch Dataset for Any-Order Masked Training, using a memory-mapped dataset.
    """
    def __init__(
        self,
        processed_dataset: torch.utils.data.Dataset,
        tokenizer: AutoTokenizer,
        mode: MaskMode,
        mask_prob: float,
        seed: int = 42,
    ):
        self.processed_dataset = processed_dataset
        self.tokenizer = tokenizer
        self.mode = mode
        self.mask_prob = mask_prob
        self.mask_token_id = tokenizer.mask_token_id
        self.base_rng = np.random.default_rng(seed)

        if self.mask_token_id is None:
            raise ValueError("Tokenizer must have a `mask_token_id`.")

    def __len__(self):
        return len(self.processed_dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self.processed_dataset[idx]
        
        # Reconstruct the TokenizedTrajectory on the fly
        unit_spans = [
            TokenizedUnit(unit_type=type, token_start=start, token_end=end, unit_index=j)
            for j, (type, start, end) in enumerate(zip(item["unit_spans_type"], item["unit_spans_start"], item["unit_spans_end"]))
        ]
        traj = TokenizedTrajectory(
            input_ids=torch.tensor(item["input_ids"]),
            unit_spans=unit_spans,
            env=item["env"],
            trajectory_id=item["id"],
        )
        
        rng = np.random.default_rng(self.base_rng.integers(1e9))
        masked_ids, loss_mask = apply_unit_mask(
            traj, self.mask_prob, self.mode, self.mask_token_id, rng
        )
        return {
            "input_ids": masked_ids,
            "target_ids": traj.input_ids.clone(),
            "loss_mask": loss_mask,
            "use_causal_mask": False,
        }

class BenchmarkUniformSampler(Sampler):
    """
    Samples elements uniformly from each benchmark, to handle imbalances.
    """
    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            if not dist.is_available():
                num_replicas = 1
            else:
                num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                rank = 0
            else:
                rank = dist.get_rank()
            
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

        # For SummarizedTrajectoryDataset, the underlying data isn't directly accessible
        # so we need to handle this gracefully.
        if hasattr(dataset, 'processed_dataset'):
             env_data = dataset.processed_dataset['env']
        else:
             # Fallback for datasets without a direct processed_dataset attribute
             # This might need adjustment depending on SummarizedTrajectoryDataset structure
             env_data = [ex.get('env', 'unknown') for ex in dataset.examples]

        self.indices_by_env = defaultdict(list)
        for idx, env in enumerate(env_data):
            self.indices_by_env[env].append(idx)

        self.env_names = list(self.indices_by_env.keys())
        self.max_len = max(len(indices) for indices in self.indices_by_env.values()) if self.indices_by_env else 0
        
        self.total_size = self.max_len * len(self.env_names)
        
        self.num_samples = self.total_size // self.num_replicas

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        
        indices = []
        for env_name in self.env_names:
            env_indices = self.indices_by_env[env_name]
            repeated_indices = (env_indices * (self.max_len // len(env_indices) + 1))[:self.max_len]
            indices.extend(repeated_indices)
        
        np.random.RandomState(self.epoch).shuffle(indices)

        if self.num_replicas > 1:
            indices = indices[self.rank:self.total_size:self.num_replicas]

        return iter(indices)

    def __len__(self):
        return self.num_samples if self.num_replicas > 1 else self.total_size

    def set_epoch(self, epoch):
        self.epoch = epoch

class SummarizedTrajectoryDataset(Dataset):
    """
    A specialized PyTorch Dataset that creates many small examples from
    each trajectory using a provided builder function. Used for Prefix-SFT.
    """
    def __init__(
        self,
        processed_dataset: torch.utils.data.Dataset,
        tokenizer: AutoTokenizer,
        build_fn: callable
    ):
        self.examples = []
        
        print(f"Building summarized trajectory dataset using {build_fn.__name__}...")
        for item in tqdm(processed_dataset):
            # Reconstruct the TokenizedTrajectory on the fly
            unit_spans = [
                TokenizedUnit(unit_type=type, token_start=start, token_end=end, unit_index=j)
                for j, (type, start, end) in enumerate(zip(item["unit_spans_type"], item["unit_spans_start"], item["unit_spans_end"]))
            ]
            traj = TokenizedTrajectory(
                input_ids=torch.tensor(item["input_ids"]),
                unit_spans=unit_spans,
                env=item["env"],
                trajectory_id=item["id"],
            )
            self.examples.extend(build_fn(traj, tokenizer))
            
        print(f"Created {len(self.examples)} training examples.")

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

    is_causal = batch["use_causal_mask"][0].item() if "use_causal_mask" in batch else False
    
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
    )
    
    loss = masked_unit_cross_entropy(outputs.logits, batch["target_ids"], batch["loss_mask"])
    return loss

# --- Main Training Loop ---

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

    # Load Config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if rank == 0:
        print(f"--- Training Config: {config_path} ---")
        print(f"Distributed: {is_distributed}, World Size: {world_size}")
        print(yaml.dump(config))
        print("--------------------")
        
        wandb = None # Initialize wandb to None
        try:
            import wandb
            wandb.init(
                project="aomt",
                name=config.get("name", "default_run"),
                config=config
            )
        except ImportError:
            print("Warning: `wandb` not installed. Skipping online logging. Please run `pip install wandb`")
            # wandb is already None


    # Setup Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model"], low_cpu_mem_usage=True)
    if tokenizer.pad_token is None: tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    if tokenizer.mask_token is None: tokenizer.add_special_tokens({'mask_token': '[MASK]'})

    # Load Data
    # Determine the absolute path to the data cache, relative to this script's location
    script_dir = os.path.dirname(__file__)
    base_data_path = os.path.abspath(os.path.join(script_dir, '..', 'data', 'processed_dataset'))
    data_path = os.path.join(base_data_path, config.get("train_split", "train"))
    
    if not os.path.exists(data_path):
         raise FileNotFoundError(f"Processed dataset not found at {data_path}. Run parse_trajectories.py first.")
    processed_dataset = load_from_disk(data_path)

    # Initialize Dataset
    mask_mode = MaskMode(config["mask_mode"])

    if mask_mode == MaskMode.PREFIX_SFT_STAGE1:
        dataset = SummarizedTrajectoryDataset(processed_dataset, tokenizer, build_prefix_sft_examples)
    elif mask_mode == MaskMode.PREFIX_SFT_STAGE2:
        dataset = SummarizedTrajectoryDataset(processed_dataset, tokenizer, build_prefix_sft_stage2_examples)
    elif mask_mode == MaskMode.STANDARD_SFT:
        dataset = SummarizedTrajectoryDataset(processed_dataset, tokenizer, build_standard_sft_examples)
    else:
        dataset = AOMTDataset(processed_dataset, tokenizer, mode=mask_mode, mask_prob=config.get("mask_prob", 0.0))
    
    # The BenchmarkUniformSampler needs access to the environment metadata.
    # For AOMTDataset, it's in `processed_dataset`. For Summarized, it's not directly available.
    # For simplicity, we'll base the sampler on the original processed_dataset for all cases.
    sampler = BenchmarkUniformSampler(processed_dataset, num_replicas=world_size, rank=rank)
    
    collator = AOMTDataCollator(tokenizer)
    dataloader = DataLoader(dataset, batch_size=config.get("per_device_batch_size", 2), collate_fn=collator, sampler=sampler, shuffle=False)

    # Initialize Model
    auto_wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=1_000_000)
    model = AutoModelForCausalLM.from_pretrained(config["model"], torch_dtype=torch.bfloat16)
    model.resize_token_embeddings(len(tokenizer))

    if is_distributed:
        model = FSDP(model, auto_wrap_policy=auto_wrap_policy, device_id=rank)
    else:
        model = model.to(rank)

    # For fair comparison, normalize training to a fixed number of steps
    # based on the original dataset size and epochs.
    # This correctly handles cases where the dataset is expanded (e.g., prefix SFT).
    steps_per_epoch = len(processed_dataset) // (config.get("per_device_batch_size", 2) * world_size)
    num_training_steps = int(steps_per_epoch * config["num_epochs"])

    if rank == 0:
        print(f"Original dataset size: {len(processed_dataset)}")
        print(f"Effective dataset size for this mode: {len(dataset)}")
        print(f"Normalizing training to {num_training_steps} total steps for fair comparison.")

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    lr_scheduler = get_scheduler(name=config.get("lr_schedule", "cosine"), optimizer=optimizer, num_warmup_steps=config.get("warmup_steps", 0), num_training_steps=num_training_steps)

    # Training Loop
    if rank == 0: print("Starting training...")
    checkpoint_dir = f"./checkpoints/{config['name']}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    dataloader_iterator = itertools.cycle(dataloader)
    progress_bar = tqdm(range(num_training_steps), desc="Training", disable=(rank != 0))

    model.train()
    for step in progress_bar:
        # Manually set epoch for the sampler if it's a new "epoch" equivalent.
        # This is an approximation to ensure sampler shuffling continues.
        if step > 0 and step % steps_per_epoch == 0:
            new_epoch = step // steps_per_epoch
            sampler.set_epoch(new_epoch)

        batch = next(dataloader_iterator)
        loss = training_step(model, batch, rank)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("gradient_clip", 1.0))
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        
        if rank == 0:
            progress_bar.set_postfix(loss=loss.item())
            if wandb:
                wandb.log({
                    "loss": loss.item(),
                    "learning_rate": lr_scheduler.get_last_lr()[0],
                    "step": step
                })

        # Checkpoint saving
        if (step + 1) % config.get("save_interval", 500) == 0:
            save_path = os.path.join(checkpoint_dir, f"step_{step+1}")
            if rank == 0: print(f"\nSaving checkpoint to {save_path}...")
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
    
    if rank == 0:
        print("\n--- Training Complete ---")
        final_save_path = os.path.join(checkpoint_dir, "final_checkpoint")
        print(f"Saving final model to {final_save_path}...")
        model.save_pretrained(final_save_path)
        tokenizer.save_pretrained(final_save_path)
        if wandb:
            wandb.finish()
        
    if is_distributed: dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the experiment config file.")
    parser.add_argument("--distributed", action='store_true', help="Flag to enable distributed training.")
    args = parser.parse_args()
    
    run_training(args.config, args.distributed)
