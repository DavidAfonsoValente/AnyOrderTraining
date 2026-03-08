"""
tasks/train_aomt.py
AOMT custom task (Action-Only, Mixed).
Verified implementation based on Engineering Specification v3.
"""

import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_scheduler
import yaml
import argparse
from tqdm import tqdm
import numpy as np
import json

# Use local dFactory/VeOmni if available
try:
    from veomni.models import build_tokenizer, build_foundation_model
except ImportError:
    def build_tokenizer(path): return AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    def build_foundation_model(**kwargs): 
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(kwargs['weights_path'], trust_remote_code=True)

# ---- Spec Section 7.2: Masking logic ------------------------------------------

def apply_unit_mask(unit_texts: list,
                    unit_types: list,
                    tokenizer,
                    mask_prob: float,
                    mode: str,
                    rng,
                    mask_token_id: int) -> tuple:
    """
    Tokenises the full trajectory flat and applies unit-level Bernoulli masking.
    """
    sep = tokenizer.eos_token_id

    all_ids, spans = [], []
    for i, (text, utype) in enumerate(zip(unit_texts, unit_types)):
        ids = tokenizer.encode(text, add_special_tokens=False)
        start = len(all_ids)
        all_ids.extend(ids)
        spans.append((start, len(all_ids), utype))
        if i < len(unit_texts) - 1:
            all_ids.append(sep)

    input_ids = torch.tensor(all_ids, dtype=torch.long)
    labels    = torch.full_like(input_ids, -100)

    masked_any = False
    for start, end, utype in spans:
        if mode == "action_only" and utype != "act":
            continue
        if rng.random() < mask_prob:
            labels[start:end] = input_ids[start:end].clone()
            input_ids[start:end] = mask_token_id
            masked_any = True

    if not masked_any:
        eligible = [(s, e, ut) for s, e, ut in spans
                    if mode == "mixed" or ut == "act"]
        if eligible:
            idx = rng.integers(len(eligible))
            s, e, _ = eligible[idx]
            labels[s:e] = input_ids[s:e].clone()
            input_ids[s:e] = mask_token_id

    return input_ids, labels


def compute_unit_mask_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over masked positions only."""
    if (labels == -100).all():
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
    )

# ---- Spec Section 7.3: AOMT Dataset class ------------------------------------

class AOMTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, mask_prob, mode, mask_token_id, seed=42):
        assert mode in ("action_only", "mixed")
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
        ex  = self.examples[idx]
        # Seed: (base_seed, index, random)
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

def run_training():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        device = f"cuda:{dist.get_local_rank()}"
        torch.cuda.set_device(device)
    else:
        rank = 0
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = build_tokenizer(config["model"]["tokenizer_path"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Setup Masking ---
    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        mask_token_id = 156895 # Fallback for LLaDA 2.0
    
    if rank == 0:
        print(f"Using Mask Token ID: {mask_token_id} ({tokenizer.decode([mask_token_id])})")

    dataset = AOMTDataset(config["data"]["train_path"], tokenizer, 
                          config["aomt"]["mask_prob"], config["aomt"]["mode"],
                          mask_token_id)
    dataloader = DataLoader(dataset, batch_size=config["train"]["per_device_batch_size"], 
                            collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
                            shuffle=True)

    model = build_foundation_model(
        weights_path=config["model"]["model_path"],
        config_path=config["model"].get("config_path"),
        torch_dtype=torch.bfloat16 if config["train"]["mixed_precision"] == "bf16" else torch.float32
    )
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["train"]["learning_rate"]), 
                                 weight_decay=config["train"]["weight_decay"])
    
    num_training_steps = len(dataloader) * config["train"]["num_epochs"]
    lr_scheduler = get_scheduler(config["train"]["lr_scheduler"], optimizer, 
                                 num_warmup_steps=config["train"]["warmup_steps"], 
                                 num_training_steps=num_training_steps)

    model.train()
    for epoch in range(config["train"]["num_epochs"]):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", disable=rank != 0)
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Bidirectional attention over all non-padding positions
            attn_mask = (input_ids != tokenizer.pad_token_id).long()

            logits = model(input_ids=input_ids, attention_mask=attn_mask).logits
            loss = compute_unit_mask_loss(logits, labels)

            loss.backward()
            
            if (pbar.n + 1) % config["train"].get("gradient_accumulation_steps", 1) == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["train"]["gradient_clip"])
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            pbar.set_postfix(loss=loss.item())

        if rank == 0:
            os.makedirs(config["train"]["output_dir"], exist_ok=True)
            model.save_pretrained(os.path.join(config["train"]["output_dir"], f"epoch_{epoch}"))

if __name__ == "__main__":
    run_training()
