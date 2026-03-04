# AOMT Engineering Specification
## Any-Order Masked Training — Implementation Guide

**For:** Engineering Team  
**Base model:** `inclusionAI/LLaDA2.0-mini` (16B/1.4B active, MoE)  
**Training framework:** dFactory (FSDP2)  
**Dataset:** `agent-eto/eto-sft-trajectory` (HuggingFace)

---

## Table of Contents

1. [Repository Structure](#1-repository-structure)
2. [Environment Setup](#2-environment-setup)
3. [Data Pipeline](#3-data-pipeline)
4. [Trajectory Unit Parser](#4-trajectory-unit-parser)
5. [Mask Sampler](#5-mask-sampler)
6. [Training Objectives: All Baselines](#6-training-objectives-all-baselines)
7. [dFactory Integration](#7-dfactory-integration)
8. [Evaluation Pipeline](#8-evaluation-pipeline)
9. [Experiment Configurations](#9-experiment-configurations)
10. [Logging and Checkpointing](#10-logging-and-checkpointing)
11. [Sanity Checks](#11-sanity-checks)
12. [Attention Mask Summary](#12-attention-mask-summary)

---

## 1. Repository Structure

```
aomt/
├── data/
│   ├── download.py             # Downloads agent-eto/eto-sft-trajectory
│   ├── parse_trajectories.py   # Converts raw dataset to internal format
│   ├── unit_parser.py          # Trajectory -> (unit, token_span) list
│   └── cache/                  # Preprocessed trajectory caches (per env)
├── training/
│   ├── mask_sampler.py         # Unit-level mask sampler (all variants)
│   ├── objectives.py           # Loss functions for all 4 baselines
│   ├── collator.py             # DataCollator: applies masks, builds inputs
│   └── trainer.py              # dFactory training loop integration
├── eval/
│   ├── task_eval.py            # Task success / average reward per benchmark
│   ├── nll_obs.py              # Observation-masked NLL metric
│   ├── nll_act.py              # Action-masked NLL metric (diagnostic)
│   └── noise_robustness.py     # Corrupted observation robustness eval
├── configs/
│   ├── sft_standard.yaml
│   ├── prefix_sft_stage1.yaml
│   ├── prefix_sft_stage2.yaml
│   ├── aomt_action_only.yaml
│   └── aomt_mixed.yaml
├── scripts/
│   ├── run_training.sh
│   ├── run_eval.sh
│   └── run_ablation_sweep.sh
└── tests/
    ├── test_unit_parser.py
    ├── test_mask_sampler.py
    └── test_loss_computation.py
```

---

## 2. Environment Setup

### 2.1 Dependencies

```bash
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.45.0 datasets accelerate

# dFactory (InclusionAI)
git clone https://github.com/inclusionAI/dFactory
cd dFactory && pip install -e .

# LLaDA 2.0 mini model
pip install huggingface_hub
huggingface-cli download inclusionAI/LLaDA2.0-mini --local-dir ./models/LLaDA2.0-mini

# Evaluation environments (not required for training)
pip install alfworld scienceworld
export ALFWORLD_DATA=./data/alfworld
```

### 2.2 Hardware Requirements

| Experiment | Minimum GPU | Recommended |
|---|---|---|
| Training (LLaDA 2.0-mini) | 4x A100 80GB | 8x A100 80GB |
| NLL-obs evaluation | 2x A100 80GB | 4x A100 80GB |
| Task evaluation | 1x A100 40GB | 2x A100 40GB |

LLaDA 2.0-mini: 16B total params, 1.4B active (MoE). With FSDP2 across 4 GPUs, peak memory per GPU is ~35GB at batch size 4, seq length 2048.

---

## 3. Data Pipeline

### 3.1 Dataset: agent-eto/eto-sft-trajectory

The ETO dataset contains expert trajectories for ALFWorld, ScienceWorld, and WebShop in ShareGPT/FastChat conversation format with GPT-4-generated rationales. This is the exact dataset used by ETO (Song et al., ACL 2024), enabling direct comparison with published numbers.

```python
from datasets import load_dataset
dataset = load_dataset("agent-eto/eto-sft-trajectory")
```

### 3.2 Raw Data Format

Conversations alternate: `human` turns are observations (O_t); `gpt` turns are actions (A_t).

```json
{
  "id": "alfworld_0",
  "conversations": [
    {"from": "human", "value": "You are in a kitchen. Task: put hot potato on countertop.\n> look"},
    {"from": "gpt",   "value": "Think: Find potato, heat it, place it.\nAct: go to fridge 1"},
    {"from": "human", "value": "You open the fridge. You see a potato 1."},
    {"from": "gpt",   "value": "Think: Found it.\nAct: take potato 1 from fridge 1"}
  ],
  "env": "alfworld"
}
```

The first `human` turn includes the task instruction + first observation. Treat it as O_0. Do not split it.

### 3.3 Internal Trajectory Format

```python
from dataclasses import dataclass
from typing import List, Literal

@dataclass
class TrajectoryUnit:
    unit_type: Literal["obs", "act"]
    text: str
    turn_index: int

@dataclass
class Trajectory:
    id: str
    env: str    # "alfworld", "sciworld", "webshop"
    units: List[TrajectoryUnit]  # strictly alternating: obs, act, obs, act, ...

def parse_conversation(example: dict) -> Trajectory:
    units = []
    for i, turn in enumerate(example["conversations"]):
        utype = "obs" if turn["from"] == "human" else "act"
        units.append(TrajectoryUnit(unit_type=utype, text=turn["value"], turn_index=i))
    
    # Validate strict alternation
    for i, u in enumerate(units):
        expected = "obs" if i % 2 == 0 else "act"
        assert u.unit_type == expected, f"Turn {i}: got {u.unit_type}, expected {expected}"
    
    return Trajectory(id=example["id"], env=example.get("env","unknown"), units=units)
```

---

## 4. Trajectory Unit Parser

Maps each trajectory into a flat token sequence and records the exact [start, end) token span for every unit. This span registry is the foundation of all masking operations.

```python
from dataclasses import dataclass
from typing import List
import torch

@dataclass
class TokenizedUnit:
    unit_type: str   # "obs" or "act"
    token_start: int # inclusive
    token_end: int   # exclusive
    unit_index: int  # position in trajectory (0 = O0, 1 = A0, 2 = O1, ...)

@dataclass
class TokenizedTrajectory:
    trajectory_id: str
    env: str
    input_ids: torch.LongTensor   # [seq_len] — clean original tokens
    unit_spans: List[TokenizedUnit]


def tokenize_trajectory(trajectory, tokenizer, max_length=2048) -> TokenizedTrajectory:
    """
    Tokenizes trajectory into a flat sequence:
      [O0 tokens] [SEP] [A0 tokens] [SEP] [O1 tokens] [SEP] ...
    
    SEP is inserted between units as a non-maskable boundary marker.
    SEP positions are never included in any unit span and never appear
    in the loss_mask.
    
    If the full sequence exceeds max_length, whole units are removed
    from the END until it fits. Never truncate mid-unit.
    """
    SEP = tokenizer.sep_token_id or tokenizer.eos_token_id
    all_ids = []
    unit_spans = []
    
    for i, unit in enumerate(trajectory.units):
        unit_ids = tokenizer.encode(unit.text, add_special_tokens=False)
        start = len(all_ids)
        all_ids.extend(unit_ids)
        end = len(all_ids)
        
        unit_spans.append(TokenizedUnit(
            unit_type=unit.unit_type,
            token_start=start,
            token_end=end,
            unit_index=i
        ))
        
        if i < len(trajectory.units) - 1:
            all_ids.append(SEP)
    
    # Truncate by removing whole units from the end
    while unit_spans and unit_spans[-1].token_end > max_length:
        removed = unit_spans.pop()
        # Also remove the SEP before this unit if present
    
    if unit_spans:
        all_ids = all_ids[:unit_spans[-1].token_end]
    
    return TokenizedTrajectory(
        trajectory_id=trajectory.id,
        env=trajectory.env,
        input_ids=torch.tensor(all_ids, dtype=torch.long),
        unit_spans=unit_spans
    )
```

---

## 5. Mask Sampler

Core AOMT component. Replaces dFactory's token-level mask sampler with a unit-level variant.

```python
import torch
import numpy as np
from enum import Enum
from typing import Tuple

class MaskMode(Enum):
    STANDARD_SFT      = "standard_sft"
    PREFIX_SFT_STAGE1 = "prefix_sft_stage1"
    ACTION_ONLY       = "action_only"
    MIXED             = "mixed"


def apply_unit_mask(
    tokenized_traj,
    mask_prob: float,
    mode: MaskMode,
    mask_token_id: int,
    rng=None,
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """
    Returns:
        masked_input_ids : [seq_len] — mask_token_id at masked unit positions
        loss_mask        : [seq_len] bool — True only at masked unit positions
    
    The loss is computed ONLY at loss_mask=True positions.
    Context positions (loss_mask=False) receive full bidirectional attention
    but do NOT contribute to the training loss.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    input_ids = tokenized_traj.input_ids.clone()
    loss_mask = torch.zeros(len(input_ids), dtype=torch.bool)
    
    units_to_mask = _select_units(tokenized_traj.unit_spans, mask_prob, mode, rng)
    
    for unit in units_to_mask:
        input_ids[unit.token_start:unit.token_end] = mask_token_id
        loss_mask[unit.token_start:unit.token_end] = True
    
    return input_ids, loss_mask


def _select_units(units, mask_prob, mode, rng):
    if mode == MaskMode.STANDARD_SFT:
        # The collator handles unpacking; here we mask all actions.
        # The causal attention mask prevents future context from leaking.
        return [u for u in units if u.unit_type == "act"]
    
    elif mode == MaskMode.PREFIX_SFT_STAGE1:
        # The collator constructs short (O_t, A_t, [MASK:O_{t+1}]) examples.
        # This function should not be called directly for stage 1.
        # See build_prefix_sft_examples() in collator.py.
        raise NotImplementedError("Use build_prefix_sft_examples() for Stage 1.")
    
    elif mode == MaskMode.ACTION_ONLY:
        return [u for u in units if u.unit_type == "act" and rng.random() < mask_prob]
    
    elif mode == MaskMode.MIXED:
        return [u for u in units if rng.random() < mask_prob]
    
    else:
        raise ValueError(f"Unknown MaskMode: {mode}")
```

**IMPORTANT — mask resampling across epochs:** The mask is resampled at `__getitem__` time using a fresh RNG seed each call. This means the same trajectory index produces a different masked sequence each time it is visited — including across epochs. This is the mechanism that produces the any-order learning effect. Do NOT cache masked sequences to disk. Always remask on the fly.

---

## 6. Training Objectives: All Baselines

### 6.1 Shared Loss Function

```python
import torch.nn.functional as F

def masked_unit_cross_entropy(logits, target_ids, loss_mask):
    """
    logits     : [B, seq_len, vocab_size]
    target_ids : [B, seq_len]  — original clean token ids
    loss_mask  : [B, seq_len]  — bool, True at masked unit positions
    
    Loss is averaged over all masked token positions across the batch.
    Positions where loss_mask=False do NOT contribute to the loss,
    even though they receive gradients through the attention mechanism.
    """
    B, S, V = logits.shape
    loss_all = F.cross_entropy(logits.view(-1, V), target_ids.view(-1), reduction="none")
    loss_masked = loss_all * loss_mask.view(-1).float()
    n_masked = loss_mask.float().sum().clamp(min=1.0)
    return loss_masked.sum() / n_masked
```

### 6.2 Prefix SFT Stage 1 Collation

```python
def build_prefix_sft_examples(tokenized_traj, tokenizer):
    """
    Constructs one training example per (O_t, A_t, O_{t+1}) triple.
    
    Input format for each example:
      [O_t tokens] [SEP] [A_t tokens] [SEP] [MASK x len(O_{t+1})]
    
    The MASK span has the SAME LENGTH as O_{t+1}. The model is told
    how many tokens to reconstruct but not their content.
    
    loss_mask is True only at the O_{t+1} positions.
    Attention: full bidirectional over the whole short sequence.
    Actions are NEVER masked in Stage 1.
    """
    SEP = tokenizer.sep_token_id or tokenizer.eos_token_id
    MASK = tokenizer.mask_token_id
    units = tokenized_traj.unit_spans
    ids = tokenized_traj.input_ids
    examples = []
    
    for i in range(len(units) - 2):
        if (units[i].unit_type == "obs" and
            units[i+1].unit_type == "act" and
            units[i+2].unit_type == "obs"):
            
            obs_t, act_t, obs_t1 = units[i], units[i+1], units[i+2]
            
            ctx = torch.cat([
                ids[obs_t.token_start:obs_t.token_end],
                torch.tensor([SEP]),
                ids[act_t.token_start:act_t.token_end],
                torch.tensor([SEP]),
            ])
            
            target_len = obs_t1.token_end - obs_t1.token_start
            masked_span = torch.full((target_len,), MASK, dtype=torch.long)
            target_span = ids[obs_t1.token_start:obs_t1.token_end]
            
            full_input = torch.cat([ctx, masked_span])
            full_target = torch.cat([ctx, target_span])  # ctx tokens are their own target (unused)
            
            loss_mask = torch.zeros(len(full_input), dtype=torch.bool)
            loss_mask[len(ctx):] = True  # Only O_{t+1} span
            
            examples.append({
                "input_ids": full_input,
                "target_ids": full_target,
                "loss_mask": loss_mask,
                "use_causal_mask": False,
            })
    
    return examples
```

---

## 7. dFactory Integration

### 7.1 Dataset Class

```python
from torch.utils.data import Dataset

class AODataset(Dataset):
    """
    On-the-fly masking: mask is resampled at __getitem__ time.
    Same trajectory produces different masked sequences across epochs.
    This is the core mechanism of AOMT.
    """
    def __init__(self, tokenized_trajectories, mode, mask_prob, mask_token_id, seed=42):
        self.trajectories = tokenized_trajectories
        self.mode = mode
        self.mask_prob = mask_prob
        self.mask_token_id = mask_token_id
        self.base_rng = np.random.default_rng(seed)
    
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        rng = np.random.default_rng(self.base_rng.integers(int(1e9)))
        
        masked_ids, loss_mask = apply_unit_mask(
            traj, self.mask_prob, self.mode, self.mask_token_id, rng
        )
        
        return {
            "input_ids": masked_ids,
            "target_ids": traj.input_ids.clone(),
            "loss_mask": loss_mask,
            "use_causal_mask": (self.mode == MaskMode.STANDARD_SFT),
        }
```

### 7.2 Training Step

```python
def training_step(model, batch, optimizer):
    input_ids = batch["input_ids"].to(model.device)
    target_ids = batch["target_ids"].to(model.device)
    loss_mask  = batch["loss_mask"].to(model.device)
    attn_mask  = batch["attention_mask"].to(model.device)
    
    if batch["use_causal_mask"]:
        # STANDARD SFT ONLY: apply causal (lower-triangular) mask
        seq_len = input_ids.shape[1]
        causal = torch.tril(torch.ones(seq_len, seq_len, device=model.device))
        combined = causal.unsqueeze(0) * attn_mask.unsqueeze(1).unsqueeze(2)
    else:
        # ALL AOMT MODES: full bidirectional attention
        combined = attn_mask.unsqueeze(1).unsqueeze(2).expand(-1, -1, input_ids.shape[1], -1)
    
    logits = model(input_ids=input_ids, attention_mask=combined).logits
    loss = masked_unit_cross_entropy(logits, target_ids, loss_mask)
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()
    
    return loss.item()
```

### 7.3 Training Defaults

```python
TRAINING_DEFAULTS = {
    "learning_rate": 2.5e-5,
    "warmup_steps": 50,
    "lr_schedule": "cosine",
    "min_lr": 2.5e-6,
    "weight_decay": 0.1,
    "gradient_clip": 1.0,
    "global_batch_size": 64,
    "per_device_batch_size": 2,
    "gradient_accumulation_steps": 8,  # 2 * 4 GPUs * 8 = 64 effective
    "max_seq_length": 2048,
    "mixed_precision": "bf16",
    "fsdp_sharding": "FULL_SHARD",
    "activation_checkpointing": True,
    "num_epochs_sft": 3,
    "num_epochs_aomt": 5,  # More epochs for AOMT; mask variety per epoch compounds
}
```

---

## 8. Evaluation Pipeline

### 8.1 Observation-Masked NLL

The primary novel metric. Computable only for models with bidirectional attention (AOMT-Action-Only and AOMT-Mixed). Cannot be computed for Standard SFT models.

```python
@torch.no_grad()
def compute_nll_obs(model, tokenized_trajectories, mask_token_id, device="cuda", batch_size=4):
    """
    For each trajectory and each observation unit O_t:
      1. Mask all tokens in O_t with [MASK]
      2. Keep ALL other units (preceding AND following) as clean context
      3. Full bidirectional forward pass
      4. Cross-entropy at O_t positions vs true O_t tokens
    
    Returns mean NLL-obs, broken down by environment and by step position t.
    Lower is better. Measures world-model quality.
    """
    model.eval()
    all_nll, nll_by_env, nll_by_position = [], {}, {}
    
    for traj in tokenized_trajectories:
        obs_units = [u for u in traj.unit_spans if u.unit_type == "obs"]
        
        for obs_unit in obs_units:
            masked_ids = traj.input_ids.clone()
            masked_ids[obs_unit.token_start:obs_unit.token_end] = mask_token_id
            
            logits = model(
                input_ids=masked_ids.unsqueeze(0).to(device),
                attention_mask=torch.ones(1, len(masked_ids), device=device),
            ).logits
            
            obs_logits = logits[0, obs_unit.token_start:obs_unit.token_end]
            obs_target = traj.input_ids[obs_unit.token_start:obs_unit.token_end].to(device)
            nll = F.cross_entropy(obs_logits, obs_target, reduction="mean").item()
            
            all_nll.append(nll)
            nll_by_env.setdefault(traj.env, []).append(nll)
            step = obs_unit.unit_index // 2
            nll_by_position.setdefault(step, []).append(nll)
    
    return {
        "mean_nll_obs": float(np.mean(all_nll)),
        "nll_by_env": {k: float(np.mean(v)) for k, v in nll_by_env.items()},
        "nll_by_position": {k: float(np.mean(v)) for k, v in sorted(nll_by_position.items())},
    }
```

### 8.2 Task Evaluation (Inference)

At inference, all AOMT models behave identically to Standard SFT: generate the next action given the current observation history. Use LLaDA 2.0's block diffusion decoding.

```python
def llada_generate(model, tokenizer, prompt_ids, gen_length=128, block_length=32, steps=32):
    """
    LLaDA 2.0 iterative unmasking for action generation.
    
    1. Append gen_length [MASK] tokens to prompt_ids
    2. Over `steps` iterations, progressively unmask tokens
       using the model's predictions (lowest-confidence tokens get remasked)
    3. Return decoded text of the unmasked response
    
    Recommended defaults: temperature=0.0, block_length=32, steps=32
    This matches the settings in LLaDA 2.0 documentation.
    """
    MASK = tokenizer.mask_token_id
    masked_resp = torch.full((1, gen_length), MASK, dtype=torch.long)
    full_input = torch.cat([prompt_ids, masked_resp], dim=1)
    prompt_len = prompt_ids.shape[1]
    
    for step_idx in range(steps):
        with torch.no_grad():
            logits = model(input_ids=full_input).logits
        
        # Get predictions and confidences for response positions
        resp_logits = logits[0, prompt_len:, :]
        probs = F.softmax(resp_logits, dim=-1)
        pred_tokens = probs.argmax(dim=-1)
        confidence = probs.max(dim=-1).values
        
        # Determine which tokens to unmask this step
        still_masked = (full_input[0, prompt_len:] == MASK)
        n_to_unmask = int(still_masked.sum() * (step_idx + 1) / steps)
        
        # Unmask highest-confidence masked positions
        masked_confidence = confidence * still_masked.float()
        _, top_indices = masked_confidence.topk(n_to_unmask)
        
        full_input[0, prompt_len + top_indices] = pred_tokens[top_indices]
    
    response_ids = full_input[0, prompt_len:]
    return tokenizer.decode(response_ids, skip_special_tokens=True)
```

### 8.3 Robustness Evaluation

```python
def corrupt_observation(text, tokenizer, corruption_frac, rng):
    """Replace corruption_frac of tokens with random vocab tokens at test time."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    n_corrupt = max(1, int(len(ids) * corruption_frac))
    positions = rng.choice(len(ids), size=n_corrupt, replace=False)
    for pos in positions:
        ids[pos] = int(rng.integers(100, tokenizer.vocab_size))
    return tokenizer.decode(ids, skip_special_tokens=True)

# Use corruption_frac in {0.1, 0.2, 0.3} at test time only.
# Wrap the environment's observation function with this before running task_eval.
```

---

## 9. Experiment Configurations

### 9.1 Standard SFT

```yaml
name: standard_sft
model: inclusionAI/LLaDA2.0-mini
use_causal_mask: true       # REQUIRED — without this it is not SFT
mask_mode: standard_sft
mask_prob: null
num_epochs: 3
learning_rate: 2.5e-5
eval_nll_obs: false         # Causal model cannot compute NLL-obs
eval_splits: [seen, unseen]
noise_fractions: [0.1, 0.2, 0.3]
```

### 9.2 Prefix SFT (Stage 1 + Stage 2)

```yaml
# Stage 1
name: prefix_sft_stage1
model: inclusionAI/LLaDA2.0-mini
use_causal_mask: false
collator: prefix_sft_stage1      # Uses build_prefix_sft_examples()
mask_mode: prefix_sft_stage1
num_epochs: 3
max_seq_length: 512              # Short single-step examples
eval_nll_obs: true
eval_task: false                 # Not deployable; eval after Stage 2
save_for_stage2: true

# Stage 2 (run after Stage 1 completes)
name: prefix_sft_stage2
model: <path_to_stage1_checkpoint>
use_causal_mask: true
mask_mode: standard_sft
num_epochs: 3
eval_task: true
eval_splits: [seen, unseen]
```

### 9.3 AOMT-Action-Only

```yaml
name: aomt_action_only
model: inclusionAI/LLaDA2.0-mini
use_causal_mask: false           # BIDIRECTIONAL — key difference from Standard SFT
mask_mode: action_only
mask_prob: 0.25
num_epochs: 5
learning_rate: 2.5e-5
eval_nll_obs: true
eval_splits: [seen, unseen]
noise_fractions: [0.1, 0.2, 0.3]
```

### 9.4 AOMT-Mixed (Full Method)

```yaml
name: aomt_mixed
model: inclusionAI/LLaDA2.0-mini
use_causal_mask: false
mask_mode: mixed
mask_prob: 0.25                  # Primary; ablation sweeps [0.15, 0.25, 0.40, 0.50]
num_epochs: 5
learning_rate: 2.5e-5
eval_nll_obs: true
eval_splits: [seen, unseen]
noise_fractions: [0.1, 0.2, 0.3]
```

### 9.5 Ablation Sweep (ALFWorld only for speed)

```bash
for p in 0.15 0.25 0.40 0.50; do
    python run_training.py \
        --config configs/aomt_mixed.yaml \
        --override mask_prob=$p \
        --override envs="[alfworld]" \
        --name aomt_mixed_p${p} \
        --output_dir ./checkpoints/ablation/p${p}
done
```

Run full benchmark (all 3 envs) only for the best-performing $p$ from the sweep.

---

## 10. Logging and Checkpointing

### 10.1 Training Log Fields (every 50 steps)

| Field | Type | Notes |
|---|---|---|
| step | int | Global step |
| epoch | float | Fractional epoch |
| loss | float | Current batch loss |
| loss_ema | float | EMA with alpha=0.99 |
| n_masked_tokens | int | Tokens masked this batch |
| n_masked_units | int | Units masked this batch |
| frac_obs_masked | float | Of masked units, fraction that were obs (MIXED only) |
| frac_act_masked | float | Of masked units, fraction that were actions |
| lr | float | Current learning rate |
| grad_norm | float | Pre-clip gradient norm |

### 10.2 Evaluation Log Fields (every 500 steps on dev set)

| Field | Type | Notes |
|---|---|---|
| nll_obs_alfworld | float | AOMT models only |
| nll_obs_sciworld | float | AOMT models only |
| nll_obs_webshop | float | AOMT models only |
| nll_act_alfworld | float | Diagnostic for all models |

### 10.3 Final Evaluation Fields

| Field | Type | Notes |
|---|---|---|
| alfworld_seen_success | float | |
| alfworld_unseen_success | float | Key generalization metric |
| sciworld_seen_score | float | |
| sciworld_unseen_score | float | Primary target: beat ETO's +22% |
| webshop_reward | float | |
| nll_obs_test | float | AOMT models only |
| nll_obs_by_position | dict | step_index -> mean NLL |
| alfworld_success_rho{01,02,03} | float | Robustness metrics |
| sciworld_score_rho{01,02,03} | float | Robustness metrics |

### 10.4 Checkpoint Schedule

Save at: every 500 steps, end of each epoch, best dev NLL-obs, best ALFWorld unseen success rate. Each checkpoint must include: model weights, optimizer state, scheduler state, RNG state, epoch/step, full config.

---

## 11. Sanity Checks

Run all of these before any full training run.

### 11.1 Unit Parser

- Given a 5-turn trajectory (O, A, O, A, O), parser produces 5 TokenizedUnits with correct types and non-overlapping spans
- Union of all unit spans + separator positions = total sequence length
- Truncation removes whole units from the end, never mid-unit
- Unit at index 0 is always "obs"; unit at index 1 is always "act"

### 11.2 Mask Sampler

- `STANDARD_SFT`: loss_mask is True only at action positions. Observations are never in loss_mask.
- `ACTION_ONLY`: loss_mask is True only at action positions. Observations are never in loss_mask.
- `MIXED`: over 1000 samples, both obs and act positions appear in loss_mask. Empirical rate should be near mask_prob.
- For any masked unit u: ALL tokens in [u.token_start, u.token_end) must equal mask_token_id. Zero partial masking.
- For any context unit u: ALL tokens in [u.token_start, u.token_end) must be unchanged from original.
- Across 10 calls to `__getitem__` on the same trajectory index (simulating 10 epochs), masks should differ on at least 8 out of 10 calls.

### 11.3 Loss Computation

- With a known loss_mask, `masked_unit_cross_entropy` equals manual mean CE over masked positions
- With loss_mask all False, function returns 0.0 or raises a clear warning (no NaN/inf)
- Gradients are non-zero only at masked unit positions

### 11.4 Attention Mask Correctness Verification

This is the most critical correctness check.

After 10 training steps:
- `aomt_action_only` loss should be LOWER than `standard_sft` loss (bidirectional context makes reconstruction easier)
- If losses are equal, the causal mask is being incorrectly applied to AOMT-Action-Only
- Run with identical data and seeds; difference should be statistically significant

### 11.5 Smoke Test Command

```bash
python run_training.py \
    --config configs/aomt_mixed.yaml \
    --smoke_test \
    --max_trajectories 5 \
    --max_steps 10 \
    --batch_size 2

# Expected: finite loss from step 1, decreasing loss, no OOM, NLL-obs is finite
```

---

## 12. Attention Mask Summary

**This table defines the single most important correctness property of the implementation.** Applying the wrong attention mask silently produces wrong gradients and invalid experiment results.

| Experiment | Attention Type | Future obs visible to masked target? | Causal mask required? |
|---|---|---|---|
| Standard SFT | **Causal (lower-triangular)** | No | **Yes** |
| Prefix SFT Stage 1 | Full bidirectional | No (only O_t, A_t provided) | No |
| Prefix SFT Stage 2 | **Causal** | No | **Yes** |
| AOMT-Action-Only | **Full bidirectional** | **Yes** | No |
| AOMT-Mixed | **Full bidirectional** | **Yes** | No |

The defining property of AOMT-Action-Only and AOMT-Mixed is that **future observations are available as context when reconstructing a masked action**. This is only possible with full bidirectional attention. If a causal mask is applied to these modes, all AOMT experiments collapse to Standard SFT behavior and the paper's core claim is untestable.

**Double-check:** For AOMT-Action-Only trained for 10 steps on the same data as Standard SFT, the loss should be measurably lower (easier task — future context available). Any identical loss values signal a causal mask bug.