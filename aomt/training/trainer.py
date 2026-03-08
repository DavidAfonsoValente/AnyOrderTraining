# aomt/training/trainer.py
import sys
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import AutoTokenizer, get_scheduler
from tqdm import tqdm
import yaml
import argparse
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
import functools
import itertools
import wandb

# --- dFactory/VeOmni Imports ---
from veomni.models import build_tokenizer, build_foundation_model

# --- Relative Imports for AOMT Package ---
from ..data.unit_parser import TokenizedTrajectory, TokenizedUnit
from .mask_sampler import MaskMode, apply_unit_mask
from .objectives import masked_unit_cross_entropy
from .collator import AOMTDataCollator, build_prefix_sft_examples, build_prefix_sft_stage2_examples, build_standard_sft_examples
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
            "use_causal_mask": self.mode == MaskMode.STANDARD_SFT,
        }

class BenchmarkUniformSampler(Sampler):
    """
    Samples elements uniformly from each benchmark, to handle imbalances.
    """
    def __init__(self, dataset: Dataset, num_replicas=None, rank=None):
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

        self.indices_by_env = defaultdict(list)
        
        if isinstance(dataset, AOMTDataset):
            env_data = dataset.processed_dataset['env']
            for idx, env in enumerate(env_data):
                self.indices_by_env[env].append(idx)
        elif isinstance(dataset, SummarizedTrajectoryDataset):
            for idx, ex in enumerate(dataset.examples):
                self.indices_by_env[ex.get('env', 'unknown')].append(idx)

        self.env_names = list(self.indices_by_env.keys())
        if not self.env_names:
            self.max_len = 0
        else:
            self.max_len = max(len(indices) for indices in self.indices_by_env.values())
        
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

def training_step(model, batch, device):
    """
    Performs a single training step, correctly handling causal vs. bidirectional attention.
    """
    # Move all tensor data to the correct device
    for key, val in batch.items():
        if isinstance(val, torch.Tensor):
            batch[key] = val.to(device)

    input_ids = batch["input_ids"]
    target_ids = batch["target_ids"]
    loss_mask = batch["loss_mask"]
    attn_mask = batch["attention_mask"] # This is the padding mask from the collator
    use_causal_mask = batch["use_causal_mask"][0].item() # Extract boolean value

    # --- Correctly construct attention mask based on mode ---
    if use_causal_mask:
        # STANDARD SFT: Apply a causal (lower-triangular) mask on top of the padding mask
        seq_len = input_ids.shape[1]
        # Create a causal mask (1s on and below the diagonal, 0s above)
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool))
        # Expand both masks to 4D for broadcasting and combine them
        # (Batch, 1, To, From) & (1, 1, To, To) -> (Batch, 1, To, From)
        # The model's attention mechanism will use this combined mask.
        # We expand the padding mask to match the shape expected by the model's attention layers.
        expanded_attn_mask = attn_mask.unsqueeze(1).unsqueeze(2).expand(-1, -1, seq_len, -1)
        final_attention_mask = expanded_attn_mask & causal_mask.unsqueeze(0).unsqueeze(0)
    else:
        # AOMT MODES: Use standard bidirectional attention, just accounting for padding.
        # We still need to expand the padding mask to the 4D shape.
        final_attention_mask = attn_mask.unsqueeze(1).unsqueeze(2)


    # Forward pass with the correctly shaped attention mask
    outputs = model(
        input_ids=input_ids,
        attention_mask=final_attention_mask,
        use_cache=False,
    )
    logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]

    loss = masked_unit_cross_entropy(logits, target_ids, loss_mask)
    return loss

def run_training(config_path: str, is_distributed: bool):
    """
    Main training function, updated to handle FSDP.
    """
    rank = 0
    world_size = 1
    if is_distributed:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(rank)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_config = config["model"]
    
    if rank == 0:
        print(f"--- Training Config: {config_path} ---")
        print(f"Distributed: {is_distributed}, World Size: {world_size}")
        print(yaml.dump(config))
        print("--------------------")
        
        try:
            import wandb
            wandb.init(project="aomt", name=config.get("name", "default_run"), config=config)
        except ImportError:
            print("Warning: `wandb` not installed. Skipping online logging.")

    tokenizer = build_tokenizer(model_config["tokenizer_path"])
    if tokenizer.pad_token is None: tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    if tokenizer.mask_token is None: tokenizer.add_special_tokens({'mask_token': '[MASK]'})

    script_dir = os.path.dirname(__file__)
    data_path = os.path.join(script_dir, '..', 'data', 'processed_dataset', config.get("train_split", "train"))
    
    if not os.path.exists(data_path):
         raise FileNotFoundError(f"Processed dataset not found at {data_path}. Run 'prepare_data.sh' first.")
    processed_dataset = load_from_disk(data_path)

    mask_mode = MaskMode(config["mask_mode"])

    if mask_mode == MaskMode.PREFIX_SFT_STAGE1:
        dataset = SummarizedTrajectoryDataset(processed_dataset, tokenizer, build_prefix_sft_examples)
    elif mask_mode == MaskMode.PREFIX_SFT_STAGE2:
        dataset = SummarizedTrajectoryDataset(processed_dataset, tokenizer, build_prefix_sft_stage2_examples)
    elif mask_mode == MaskMode.STANDARD_SFT:
        dataset = SummarizedTrajectoryDataset(processed_dataset, tokenizer, build_standard_sft_examples)
    else:
        dataset = AOMTDataset(processed_dataset, tokenizer, mode=mask_mode, mask_prob=config.get("mask_prob", 0.0))
    
    sampler = BenchmarkUniformSampler(dataset, num_replicas=world_size, rank=rank)
    
    collator = AOMTDataCollator(tokenizer)
    dataloader = DataLoader(dataset, batch_size=config.get("per_device_batch_size", 2), collate_fn=collator, sampler=sampler, shuffle=False)

    auto_wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=1_000_000)
    
    model = build_foundation_model(
        config_path=model_config["config_path"],
        weights_path=model_config["model_path"],
        torch_dtype=torch.bfloat16 if config.get("mixed_precision") == "bf16" else torch.float32, 
        attn_implementation=model_config.get("attn_implementation", "sdpa"),
    )
    model.resize_token_embeddings(len(tokenizer))

    if is_distributed:
        model = FSDP(model, auto_wrap_policy=auto_wrap_policy, device_id=rank)
    else:
        model = model.to(rank)

    steps_per_epoch = len(dataset) // (config.get("global_batch_size", 64))
    num_training_steps = int(steps_per_epoch * config["num_epochs"])

    if rank == 0:
        print(f"Original dataset size: {len(processed_dataset)}")
        print(f"Effective dataset size for this mode: {len(dataset)}")
        print(f"Normalizing training to {num_training_steps} total steps.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    lr_scheduler = get_scheduler(name=config.get("lr_schedule", "cosine"), optimizer=optimizer, num_warmup_steps=config.get("warmup_steps", 0), num_training_steps=num_training_steps)

    if rank == 0: print("Starting training...")
    checkpoint_dir = f"./checkpoints/{config['name']}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    dataloader_iterator = itertools.cycle(dataloader)
    progress_bar = tqdm(range(num_training_steps), desc="Training", disable=(rank != 0))

    model.train()
    for step in progress_bar:
        if step > 0 and step % steps_per_epoch == 0:
            sampler.set_epoch(step // steps_per_epoch)

        batch = next(dataloader_iterator)
        loss = training_step(model, batch, rank)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("gradient_clip", 1.0))
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        
        if rank == 0:
            progress_bar.set_postfix(loss=loss.item())
            if 'wandb' in locals():
                wandb.log({"loss": loss.item(), "learning_rate": lr_scheduler.get_last_lr()[0], "step": step})

    if rank == 0:
        print("\n--- Training Complete ---")
        if 'wandb' in locals():
            wandb.finish()
        
    if is_distributed: dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the experiment config file.")
    parser.add_argument("--distributed", action='store_true', help="Flag to enable distributed training.")
    args = parser.parse_args()
    
    run_training(args.config, args.distributed)
