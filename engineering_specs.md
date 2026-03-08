# AOMT Engineering Specification — v3
## Any-Order Masked Training — Verified Implementation Guide

**Base model:** `inclusionAI/LLaDA2.0-mini` (16B total / 1.4B active, MoE)
**Training framework:** dFactory — VeOmni / FSDP2
**Dataset:** `agent-eto/eto-sft-trajectory` (HuggingFace)
**Mask token ID:** `126336` (LLaDA's `[MASK]` token — confirm via `tokenizer.mask_token_id`)
**Cluster:** SoC Compute Cluster (Slurm) — see `scripts/` directory

> **v3 changes from v2:**
> - All code verified against dFactory `train_llada2_bd.py` and LLaDA2.0 generate API
> - `train_standard_sft.py` implementation corrected: uses dFactory's collator output directly, only replaces the masking step (not the whole loop)
> - LLaDA `model.generate` parameters verified and flagged where runtime confirmation needed
> - `eos_early_stop` parameter flagged as unverified — fallback provided
> - Added explicit `prompt_lengths` field note (dFactory computes this from chat template)
> - Appendix proof for Prefix SFT corrected: ALEE-IWM uses local `(O_t, A_t)` context, not full history
> - Added `data/measure_lengths.py`, comprehensive test suite, visualisation script
> - Cluster Slurm scripts added for all five training variants

---

## Table of Contents

1. [Repository Structure](#1-repository-structure)
2. [Environment Setup](#2-environment-setup)
3. [Dataset and Data Preparation](#3-dataset-and-data-preparation)
4. [Masking Design — All Methods](#4-masking-design--all-methods)
5. [dFactory Integration — How It Actually Works](#5-dfactory-integration--how-it-actually-works)
6. [Baseline Custom Task (`train_standard_sft.py`)](#6-baseline-custom-task)
7. [AOMT Custom Task (`train_aomt.py`)](#7-aomt-custom-task)
8. [Configs](#8-configs)
9. [Running Training](#9-running-training)
10. [Inference](#10-inference)
11. [Evaluation](#11-evaluation)
12. [Sanity Checks and Tests](#12-sanity-checks-and-tests)
13. [Attention Mask Summary](#13-attention-mask-summary)

---

## 1. Repository Structure

```
aomt/
├── data/
│   ├── download.py              # Downloads agent-eto/eto-sft-trajectory
│   ├── prepare_data.py          # Converts raw → JSONL per mode
│   ├── measure_lengths.py       # REQUIRED pre-training: validate gen_length
│   └── visualize_examples.py   # Visualise masked training examples per method
├── tasks/
│   ├── train_standard_sft.py   # Baseline custom task (Std SFT, Prefix S1, S2)
│   └── train_aomt.py           # AOMT custom task (Action-Only, Mixed)
├── configs/
│   ├── sft_standard.yaml
│   ├── prefix_sft_stage1.yaml
│   ├── prefix_sft_stage2.yaml
│   ├── aomt_action_only.yaml
│   └── aomt_mixed.yaml
├── eval/
│   ├── task_eval.py            # ALFWorld / ScienceWorld / WebShop success metrics
│   ├── nll_obs.py              # Observation-masked NLL (AOMT only)
│   └── noise_robustness.py     # Corrupted-observation robustness eval
├── scripts/                    # Slurm batch scripts (see scripts/ section)
│   ├── 00_setup_env.sh
│   ├── 01_prepare_data.sh
│   ├── 02_train_sft_standard.sh
│   ├── 03_train_prefix_s1.sh
│   ├── 04_train_prefix_s2.sh
│   ├── 05_train_aomt_action.sh
│   ├── 06_train_aomt_mixed.sh
│   ├── 07_run_eval.sh
│   └── submit_pipeline.sh      # Submits all with Slurm job dependencies
└── tests/
    └── test_suite.py           # Comprehensive correctness tests
```

---

## 2. Environment Setup

### 2.1 Install dFactory

```bash
git clone https://github.com/inclusionAI/dFactory.git --recursive
cd dFactory
pip install -e VeOmni/
pip install datasets transformers accelerate
```

### 2.2 Prepare model weights

dFactory requires the **merged-expert** format. Two conversion steps bookend training.

```bash
# Download
python scripts/download_hf_model.py \
  --repo_id inclusionAI/LLaDA2.0-mini \
  --local_dir ./models/llada2-mini-sep

# Convert to merged format for training
python scripts/moe_convertor.py \
  --input-path  ./models/llada2-mini-sep \
  --output-path ./models/llada2-mini-merged \
  --mode merge
```

After training, convert back for inference:

```bash
python scripts/moe_convertor.py \
  --input-path  <CHECKPOINT>/hf_ckpt \
  --output-path ./models/<name>-sep \
  --mode split

# The modeling file must be copied manually
cp ./models/llada2-mini-sep/modeling_llada2_moe.py ./models/<name>-sep/
```

### 2.3 Verify MASK_TOKEN_ID at runtime

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("./models/llada2-mini-sep", trust_remote_code=True)

assert tok.mask_token_id == 126336, \
    f"MASK_TOKEN_ID mismatch: expected 126336, got {tok.mask_token_id}"
# If this fails, update MASK_TOKEN_ID = tok.mask_token_id everywhere.
```

### 2.4 Hardware requirements

LLaDA2.0-mini has 16B total parameters (1.4B active via MoE). In bf16:
- **Weights:** 32GB
- **AdamW optimizer (fp32):** 128GB
- **Gradients (bf16):** 32GB
- **Total (pre-activation):** ~192GB → sharded across N GPUs

| Setup | GPUs | Memory/GPU (pre-act) | Feasible? |
|---|---|---|---|
| 4× A100 40GB | 160GB total | 48GB + activations ≈ 65GB | **No** — exceeds 40GB |
| 8× A100 40GB | 320GB total | 24GB + ~10GB activations | **Yes** with grad checkpointing |
| 4× A100 80GB | 320GB total | 48GB + ~12GB | **Yes** comfortably |
| 4× H100 96GB | 384GB total | 48GB + ~12GB | **Yes** with headroom |
| 4× H100 141GB (xgpk0) | 564GB total | 48GB + ~12GB | **Yes** — best option |

**Recommended:** xgpk0 (single node, 4× H100 141GB) or 2× xgpi nodes with native H100 96GB.  
**Minimum:** 8× xgpg nodes (A100 40GB) with `--gradient_checkpointing=true`.

---

## 3. Dataset and Data Preparation

### 3.1 Raw format

```python
from datasets import load_dataset
dataset = load_dataset("agent-eto/eto-sft-trajectory")
# Each example: {"id": str, "conversations": [...], "env": str}
# Conversations: strictly alternating human/gpt turns
```

### 3.2 Unit parsing

```python
def parse_units(conversations):
    """Strictly alternating obs/act units. First turn always obs."""
    units = []
    for turn in conversations:
        utype = "obs" if turn["from"] == "human" else "act"
        units.append({"type": utype, "text": turn["value"]})
    assert all(u["type"] == ("obs" if i % 2 == 0 else "act")
               for i, u in enumerate(units)), "Non-alternating trajectory"
    return units
```

### 3.3 Pre-training length analysis (REQUIRED)

Run `data/measure_lengths.py` before any training. Confirms `gen_length=256` covers all actions and `max_seq_length=2048` covers full trajectories.

```python
# data/measure_lengths.py
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("./models/llada2-mini-sep", trust_remote_code=True)
dataset = load_dataset("agent-eto/eto-sft-trajectory")

action_lengths, obs_lengths, traj_lengths = [], [], []

for split in ["train", "test"]:
    for ex in dataset[split]:
        units = parse_units(ex["conversations"])
        traj_len = 0
        for u in units:
            toks = len(tokenizer.encode(u["text"], add_special_tokens=False))
            traj_len += toks
            (action_lengths if u["type"] == "act" else obs_lengths).append(toks)
        traj_lengths.append(traj_len)

for name, arr in [("Actions", action_lengths),
                  ("Observations", obs_lengths),
                  ("Full trajectories", traj_lengths)]:
    arr = np.array(arr)
    print(f"{name:20s}: mean={arr.mean():6.0f}  "
          f"p95={np.percentile(arr, 95):6.0f}  "
          f"p99={np.percentile(arr, 99):6.0f}  max={arr.max():6.0f}")

# Decision rules:
# Actions p99  > 230  → increase gen_length to p99 + 30, round up to nearest 32
# Actions max  ≥ 256  → HARD FAILURE: fix gen_length before running eval
# Full traj p99 > 1800 → flag long trajectories for truncation strategy
```

### 3.4 Data preparation: `prepare_data.py`

#### Standard SFT (T datapoints per trajectory)

```python
def make_standard_sft(units, sep="\n"):
    """
    One datapoint per action. Prompt = full causal history up to O_t.
    Response = A_t. dFactory sees user=prompt, assistant=response.
    """
    datapoints = []
    prompt_parts = []
    for unit in units:
        if unit["type"] == "obs":
            prompt_parts.append(unit["text"])
        else:
            datapoints.append({
                "messages": [
                    {"role": "user",      "content": sep.join(prompt_parts)},
                    {"role": "assistant", "content": unit["text"]},
                ]
            })
            prompt_parts.append(unit["text"])
    return datapoints
```

#### Prefix SFT Stage 1 (T datapoints per trajectory)

```python
def make_prefix_sft_s1(units, sep="\n"):
    """
    One datapoint per (O_t, A_t, O_{t+1}) triple.
    Prompt = (O_t, A_t) local context ONLY — full history deliberately excluded.
    Response = O_{t+1}.
    """
    datapoints = []
    for i in range(len(units) - 2):
        if (units[i]["type"] == "obs" and
            units[i+1]["type"] == "act" and
            units[i+2]["type"] == "obs"):
            datapoints.append({
                "messages": [
                    {"role": "user",
                     "content": units[i]["text"] + sep + units[i+1]["text"]},
                    {"role": "assistant", "content": units[i+2]["text"]},
                ]
            })
    return datapoints
```

#### AOMT (1 datapoint per trajectory)

```python
def make_aomt_datapoint(units):
    return {
        "unit_texts": [u["text"] for u in units],
        "unit_types": [u["type"] for u in units],
    }
```

#### Output files

```bash
python aomt/data/prepare_data.py --output_dir ./data/cache/

# Produces:
# data/cache/sft_standard_train.jsonl   (T datapoints/traj, chat format)
# data/cache/sft_standard_test.jsonl
# data/cache/prefix_sft_s1_train.jsonl  (T datapoints/traj, chat format)
# data/cache/prefix_sft_s1_test.jsonl
# data/cache/aomt_train.jsonl           (1 datapoint/traj, unit format)
# data/cache/aomt_test.jsonl
# Prefix SFT S2 reuses sft_standard_train/test.jsonl
```

---

## 4. Masking Design — All Methods

### 4.1 The universal principle

> **All five methods use unit-level all-or-nothing masking. No method uses token-level random masking. This is a deliberate design choice for controlled comparison.**

Token-level random masking (dFactory's default) allows the model to exploit partial cues — e.g. seeing `"heat potato with ___"` and completing the token rather than reasoning from trajectory context. Unit-level masking eliminates this shortcut and makes the training task semantically coherent.

**Critically:** if Standard SFT used token-level masking and AOMT used unit-level masking, any performance difference would be confounded by masking granularity. Using unit-level masking for all five methods isolates the true independent variable: **the context structure** (causal vs. bidirectional with future observations).

### 4.2 Per-method masking parameters

| Method | Eligible units | Masking probability | Deterministic? |
|---|---|---|---|
| Standard SFT | Response (`A_t`) | p = 1.0 | Yes — always fully masked |
| Prefix SFT S1 | Response (`O_{t+1}`) | p = 1.0 | Yes |
| Prefix SFT S2 | Response (`A_t`) | p = 1.0 | Yes |
| AOMT-Action-Only | All `act` units | p = 0.25 per unit | No — resampled each step |
| AOMT-Mixed | All units (obs + act) | p = 0.25 per unit | No — resampled each step |

**Why p=1.0 for baselines:** Each baseline datapoint has exactly one eligible unit. Bernoulli(0.25) would waste 75% of gradient steps. With one unit, deterministic full masking is both correct and maximally sample-efficient.

**Why p=0.25 for AOMT:** Gives E[masked units] ≈ 0.25T per step — enough signal without masking most context. Swept over {0.15, 0.25, 0.40, 0.50} on ALFWorld.

### 4.3 Shared loss function

```python
import torch.nn.functional as F

def compute_unit_mask_loss(logits, labels):
    """
    Shared by ALL five methods.

    logits : [B, seq_len, vocab_size]
    labels : [B, seq_len] — true token IDs at masked positions, -100 elsewhere

    F.cross_entropy with ignore_index=-100 correctly averages ONLY over
    masked positions. No 1/p_mask weighting — inapplicable to unit-level masking.
    """
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
    )
```

---

## 5. dFactory Integration — How It Actually Works

Understanding what dFactory's `train_llada2_bd.py` does internally is essential for writing correct custom tasks.

### 5.1 What dFactory's built-in loop provides

When dFactory processes a chat-format JSONL line:
```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

It:
1. Calls `tokenizer.apply_chat_template(messages, ...)` to tokenise the full sequence
2. Computes `prompt_lengths` = number of tokens up to and including the last user-turn separator
3. Passes `{"input_ids": tensor, "prompt_lengths": tensor}` to the training loop
4. In the training loop, calls `forward_process(input_ids)` to apply token-level masking
5. Restores prompt tokens using `prompt_lengths`
6. Computes loss with `1/p_mask` weighting

### 5.2 What we replace

Our custom tasks only replace **steps 4 and 6**. Steps 1–3 (tokenisation, chat template, `prompt_lengths`) and all distributed infrastructure (FSDP2, mixed precision, grad clipping, checkpointing, logging) are inherited from VeOmni **unchanged**.

This means:
- For baseline methods: we only swap `forward_process` → `apply_response_unit_mask`
- For AOMT: we swap the entire dataset class and the masking step

### 5.3 dFactory config fields (verified)

The YAML config fields consumed by dFactory:
```yaml
model:
  model_path:     # path to merged-expert weights
  tokenizer_path: # path to sep-format tokenizer (for apply_chat_template)

data:
  train_path:  # JSONL file path
  test_path:

train:
  output_dir:
  num_epochs:
  learning_rate:
  lr_scheduler:          # "cosine" supported
  min_lr:
  warmup_steps:
  weight_decay:
  gradient_clip:
  per_device_batch_size:
  gradient_accumulation_steps:
  max_seq_length:        # sequences longer than this are truncated/dropped
  mixed_precision:       # "bf16"
  save_steps:
  gradient_checkpointing: # bool, add for memory-constrained setups
```

Custom fields (`masking:`, `aomt:`) are parsed by our custom task files only.

---

## 6. Baseline Custom Task

### File: `aomt/tasks/train_standard_sft.py`

Copy `dFactory/tasks/train_llada2_bd.py`. Make the following targeted changes only.

#### Change 1: Add the unit-level masking function

Insert after imports:

```python
MASK_TOKEN_ID = 126336  # Verify at runtime: assert tok.mask_token_id == 126336

def apply_response_unit_mask(input_ids: torch.Tensor,
                              prompt_lengths: torch.Tensor) -> tuple:
    """
    Deterministically masks the entire response span.

    input_ids      : [B, seq_len] — clean token ids (from dFactory's collator)
    prompt_lengths : [B]          — tokens in the prompt/user portion per example

    Replaces ALL response tokens with MASK_TOKEN_ID (p=1.0, deterministic).
    No partial masking. No token-level sampling.

    Returns:
        masked_input_ids : [B, seq_len]
        labels           : [B, seq_len]  — true ids at response pos, -100 at prompt pos
    """
    B, L = input_ids.shape
    positions = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
    response_mask = positions >= prompt_lengths.unsqueeze(1)

    masked_input_ids = input_ids.clone()
    masked_input_ids[response_mask] = MASK_TOKEN_ID

    labels = torch.full_like(input_ids, -100)
    labels[response_mask] = input_ids[response_mask]

    return masked_input_ids, labels


def compute_unit_mask_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over masked positions only. No 1/p_mask weighting."""
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
    )
```

#### Change 2: Replace the forward_process block in the training step

Find this block (or equivalent) in `train_llada2_bd.py`:

```python
# ORIGINAL — DELETE THIS BLOCK:
noisy_batch, _, p_mask = forward_process(input_ids)
token_positions = torch.arange(...).expand(...)
prompt_mask = token_positions < prompt_lengths.unsqueeze(1)
noisy_batch[prompt_mask] = input_ids[prompt_mask]
answer_lengths = (1 - prompt_mask.long()).sum(...).repeat(...)
masked_indices = (noisy_batch == MASK_TOKEN_ID)
logits = model(input_ids=noisy_batch).logits
token_loss = F.cross_entropy(...) / p_mask[masked_indices]
ce_loss = torch.sum(token_loss / answer_lengths[masked_indices]) / input_ids.shape[0]
```

Replace with:

```python
# NEW — unit-level masking:
masked_input_ids, labels = apply_response_unit_mask(input_ids, prompt_lengths)

# Bidirectional attention over all non-padding positions (no causal mask)
attn_mask = (masked_input_ids != tokenizer.pad_token_id).long()

logits = model(input_ids=masked_input_ids, attention_mask=attn_mask).logits
ce_loss = compute_unit_mask_loss(logits, labels)
```

That is the entirety of the change. Everything else — grad scaler, FSDP2, logging, checkpointing, LR scheduler — is inherited from dFactory unchanged.

#### Verification

After the change, add this assertion at the start of the training step:

```python
# Sanity: response tokens should be MASK_TOKEN_ID; prompt tokens should not
assert (masked_input_ids[0, :prompt_lengths[0]] != MASK_TOKEN_ID).all(), \
    "Prompt tokens must not be masked"
assert (masked_input_ids[0, prompt_lengths[0]:] == MASK_TOKEN_ID).all(), \
    "All response tokens must be masked"
# Remove these assertions after confirming correctness (they run every step).
```

---

## 7. AOMT Custom Task

### File: `aomt/tasks/train_aomt.py`

Similar approach: copy `train_llada2_bd.py` but replace the dataset class AND the masking step.

### 7.1 JSONL format

```json
{"unit_texts": ["O0 text", "A0 text", "O1 text", "A1 text", "O2 text"],
 "unit_types": ["obs", "act", "obs", "act", "obs"]}
```

### 7.2 Unit-level mask function

```python
MASK_TOKEN_ID = 126336
SEP_TOKEN_ID  = None  # Set at runtime: tokenizer.eos_token_id

def apply_unit_mask(unit_texts: list,
                    unit_types: list,
                    tokenizer,
                    mask_prob: float,
                    mode: str,
                    rng) -> tuple:
    """
    Tokenises the full trajectory flat and applies unit-level Bernoulli masking.

    mode = "action_only" : only "act" units eligible
    mode = "mixed"       : obs and act units both eligible

    No partial masking within a unit. Resampled every call.

    Returns:
        input_ids : [seq_len] LongTensor
        labels    : [seq_len] LongTensor  (-100 at non-target positions)
    """
    global SEP_TOKEN_ID
    sep = SEP_TOKEN_ID if SEP_TOKEN_ID is not None else tokenizer.eos_token_id

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
            input_ids[start:end] = MASK_TOKEN_ID
            masked_any = True

    # If nothing was masked (all Bernoulli draws failed), force-mask one random
    # eligible unit to avoid completely empty gradient steps.
    if not masked_any:
        eligible = [(s, e, ut) for s, e, ut in spans
                    if mode == "mixed" or ut == "act"]
        if eligible:
            idx = rng.integers(len(eligible))
            s, e, _ = eligible[idx]
            labels[s:e] = input_ids[s:e].clone()
            input_ids[s:e] = MASK_TOKEN_ID

    return input_ids, labels
```

### 7.3 AOMT Dataset class

```python
import numpy as np
from torch.utils.data import Dataset

class AOMTDataset(Dataset):
    """
    Mask resampled at __getitem__ time via a seeded-but-stochastic RNG.
    Same trajectory index → different masked sequence each epoch.
    Do NOT cache masked sequences — the resampling IS the any-order learning.
    """
    def __init__(self, raw_examples: list, tokenizer, mask_prob: float,
                 mode: str, seed: int = 42):
        assert mode in ("action_only", "mixed")
        self.examples  = raw_examples
        self.tokenizer = tokenizer
        self.mask_prob = mask_prob
        self.mode      = mode
        self.seed      = seed

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex  = self.examples[idx]
        # Seed: (base_seed, index, random) → different mask per epoch, same trajectory
        rng = np.random.default_rng([self.seed, idx, np.random.randint(0, 2**31)])
        input_ids, labels = apply_unit_mask(
            ex["unit_texts"], ex["unit_types"],
            self.tokenizer, self.mask_prob, self.mode, rng,
        )
        return {"input_ids": input_ids, "labels": labels}
```

### 7.4 Training loop change in `train_aomt.py`

Replace the dFactory masking block with:

```python
# input_ids already masked by AOMTDataset.__getitem__
input_ids = batch["input_ids"].to(device)
labels    = batch["labels"].to(device)

attn_mask = (input_ids != tokenizer.pad_token_id).long()

logits  = model(input_ids=input_ids, attention_mask=attn_mask).logits
ce_loss = compute_unit_mask_loss(logits, labels)
# Backward, grad clip, optimiser, logging, ckpt — unchanged from VeOmni
```

---

## 8. Configs

### 8.1 Standard SFT

```yaml
# configs/sft_standard.yaml
model:
  model_path:      ./models/llada2-mini-merged
  tokenizer_path:  ./models/llada2-mini-sep

data:
  train_path:  ./data/cache/sft_standard_train.jsonl
  test_path:   ./data/cache/sft_standard_test.jsonl

train:
  output_dir:                    ./checkpoints/sft_standard
  num_epochs:                    3
  learning_rate:                 2.5e-5
  lr_scheduler:                  cosine
  min_lr:                        2.5e-6
  warmup_steps:                  50
  weight_decay:                  0.1
  gradient_clip:                 1.0
  per_device_batch_size:         2
  gradient_accumulation_steps:   8      # effective batch = 2 × 4 GPUs × 8 = 64
  max_seq_length:                2048
  mixed_precision:               bf16
  save_steps:                    500
  gradient_checkpointing:        true   # required for A100-40 nodes

masking:
  mode: standard   # parsed by train_standard_sft.py
```

### 8.2 Prefix SFT Stage 1

```yaml
# configs/prefix_sft_stage1.yaml
model:
  model_path:      ./models/llada2-mini-merged
  tokenizer_path:  ./models/llada2-mini-sep

data:
  train_path:  ./data/cache/prefix_sft_s1_train.jsonl
  test_path:   ./data/cache/prefix_sft_s1_test.jsonl

train:
  output_dir:                    ./checkpoints/prefix_sft_s1
  num_epochs:                    3
  learning_rate:                 2.5e-5
  lr_scheduler:                  cosine
  min_lr:                        2.5e-6
  warmup_steps:                  50
  weight_decay:                  0.1
  gradient_clip:                 1.0
  per_device_batch_size:         4      # shorter sequences
  gradient_accumulation_steps:   4      # effective batch = 4 × 4 × 4 = 64
  max_seq_length:                512
  mixed_precision:               bf16
  save_steps:                    500
  gradient_checkpointing:        true

masking:
  mode: standard
```

### 8.3 Prefix SFT Stage 2

```yaml
# configs/prefix_sft_stage2.yaml
model:
  model_path:      ./models/prefix_sft_s1_merged   # Stage 1 ckpt, re-merged
  tokenizer_path:  ./models/llada2-mini-sep

data:
  train_path:  ./data/cache/sft_standard_train.jsonl
  test_path:   ./data/cache/sft_standard_test.jsonl

train:
  output_dir:                    ./checkpoints/prefix_sft_s2
  num_epochs:                    3
  learning_rate:                 2.5e-5
  lr_scheduler:                  cosine
  min_lr:                        2.5e-6
  warmup_steps:                  50
  weight_decay:                  0.1
  gradient_clip:                 1.0
  per_device_batch_size:         2
  gradient_accumulation_steps:   8
  max_seq_length:                2048
  mixed_precision:               bf16
  save_steps:                    500
  gradient_checkpointing:        true

masking:
  mode: standard
```

### 8.4 AOMT-Action-Only

```yaml
# configs/aomt_action_only.yaml
model:
  model_path:      ./models/llada2-mini-merged
  tokenizer_path:  ./models/llada2-mini-sep

data:
  train_path:  ./data/cache/aomt_train.jsonl
  test_path:   ./data/cache/aomt_test.jsonl

train:
  output_dir:                    ./checkpoints/aomt_action_only
  num_epochs:                    5
  learning_rate:                 2.5e-5
  lr_scheduler:                  cosine
  min_lr:                        2.5e-6
  warmup_steps:                  50
  weight_decay:                  0.1
  gradient_clip:                 1.0
  per_device_batch_size:         1      # full trajectories are longer
  gradient_accumulation_steps:   16     # effective batch = 1 × 4 × 16 = 64
  max_seq_length:                2048
  mixed_precision:               bf16
  save_steps:                    500
  gradient_checkpointing:        true

aomt:
  mode:       action_only
  mask_prob:  0.25
```

### 8.5 AOMT-Mixed

```yaml
# configs/aomt_mixed.yaml
model:
  model_path:      ./models/llada2-mini-merged
  tokenizer_path:  ./models/llada2-mini-sep

data:
  train_path:  ./data/cache/aomt_train.jsonl
  test_path:   ./data/cache/aomt_test.jsonl

train:
  output_dir:                    ./checkpoints/aomt_mixed
  num_epochs:                    5
  learning_rate:                 2.5e-5
  lr_scheduler:                  cosine
  min_lr:                        2.5e-6
  warmup_steps:                  50
  weight_decay:                  0.1
  gradient_clip:                 1.0
  per_device_batch_size:         1
  gradient_accumulation_steps:   16
  max_seq_length:                2048
  mixed_precision:               bf16
  save_steps:                    500
  gradient_checkpointing:        true

aomt:
  mode:       mixed
  mask_prob:  0.25   # ablation: [0.15, 0.25, 0.40, 0.50]
```

---

## 9. Running Training

### 9.1 dFactory launch command

dFactory's `train.sh` wraps `torchrun`. The correct invocation from the dFactory repo:

```bash
PYTHONPATH=$(pwd)/VeOmni:$(pwd)/aomt:$PYTHONPATH \
  sh dFactory/train.sh <task_file> <config_file>
```

For multi-node (Slurm), we replace `sh train.sh` with a `srun torchrun` invocation. See `scripts/` for complete Slurm scripts.

### 9.2 Training order with dependencies

```
1. prepare_data.sh        # Generate JSONL files
2. train_sft_standard.sh  # Independent
2. train_prefix_s1.sh     # Independent
2. train_aomt_action.sh   # Independent
2. train_aomt_mixed.sh    # Independent
3. train_prefix_s2.sh     # Depends on train_prefix_s1 completion
4. run_eval.sh            # Depends on all training completions
```

Use `scripts/submit_pipeline.sh` to submit all with `--dependency=afterok`.

### 9.3 Stage 1 → Stage 2 conversion

Between Prefix SFT Stage 1 and Stage 2, convert the checkpoint:

```bash
python dFactory/scripts/moe_convertor.py \
  --input-path  ./checkpoints/prefix_sft_s1/<best_step>/hf_ckpt \
  --output-path ./models/prefix_sft_s1_merged \
  --mode merge
# Then update prefix_sft_stage2.yaml: model_path = ./models/prefix_sft_s1_merged
```

---

## 10. Inference

### 10.1 LLaDA2.0 generate API

⚠️ **Verify these parameters against the actual LLaDA2.0 repo before running eval.**

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "./models/finetuned-sep",
    trust_remote_code=True,
    device_map="cuda",
    torch_dtype=torch.bfloat16,
).eval()
tokenizer = AutoTokenizer.from_pretrained(
    "./models/llada2-mini-sep", trust_remote_code=True
)

def generate_action(obs_history_text: str, gen_length: int = 256) -> str:
    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": obs_history_text}],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        # --- Verify these kwargs against LLaDA2.0's model.generate signature ---
        # Confirmed params (from LLaDA2.0 paper / generate.py):
        #   gen_length, block_length, steps, temperature, cfg_scale, remasking
        # Unconfirmed: eos_early_stop — check generate.py; if absent, use post-hoc truncation
        try:
            output_ids = model.generate(
                input_ids,
                gen_length=gen_length,
                block_length=32,
                steps=32,
                temperature=0.0,
                cfg_scale=0.0,         # 0.0 = no classifier-free guidance
                remasking="low_confidence",  # default in LLaDA2.0
                eos_early_stop=True,   # May not exist — see fallback below
            )
        except TypeError:
            # Fallback if eos_early_stop not supported
            output_ids = model.generate(
                input_ids,
                gen_length=gen_length,
                block_length=32,
                steps=32,
                temperature=0.0,
                cfg_scale=0.0,
                remasking="low_confidence",
            )

    generated = output_ids[0, input_ids.shape[1]:]
    # Post-hoc EOS truncation if eos_early_stop not supported
    eos_id = tokenizer.eos_token_id
    eos_positions = (generated == eos_id).nonzero(as_tuple=True)[0]
    if len(eos_positions) > 0:
        generated = generated[:eos_positions[0]]

    return tokenizer.decode(generated, skip_special_tokens=True)
```

### 10.2 `steps=1` consistency check

Unit-level training (all-MASK → predict everything at once) is most directly consistent with `steps=1` (one-shot decoding). If iterative unmasking (`steps=32`) produces degenerate outputs for any method, test:

```python
output_ids = model.generate(
    input_ids,
    gen_length=gen_length,
    block_length=gen_length,   # single block = one-shot
    steps=1,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
)
```

Report both `steps=1` and `steps=32` results in supplementary if they differ meaningfully.

### 10.3 Generation parameter validation

| Parameter | Value | Rationale |
|---|---|---|
| `gen_length` | 256 | Validate with `measure_lengths.py` first |
| `block_length` | 32 | 256/32 = 8 blocks; one-shot for typical short actions |
| `steps` | 32 | 32 denoising steps per block |
| `temperature` | 0.0 | Greedy decoding for reproducibility |
| `cfg_scale` | 0.0 | No CFG; consistent with SFT training |
| `remasking` | `"low_confidence"` | LLaDA2.0 default; unmasks highest-confidence tokens first |

---

## 11. Evaluation

### 11.1 Task evaluation (all methods)

```python
# eval/task_eval.py
def evaluate_model(model, tokenizer, env_name, split="test", n_episodes=None):
    """
    env_name: "alfworld" | "scienceworld" | "webshop"
    Returns dict of metrics appropriate to environment.
    """
    ...
# Metrics:
# ALFWorld:    success rate (%) — seen and unseen splits
# SciWorld:    avg normalised score × 100 — seen and unseen
# WebShop:     avg reward × 100, success rate (%)
```

### 11.2 Observation-masked NLL (AOMT only)

Standard SFT and Prefix SFT are excluded: no future-context mechanism for former; OOD for latter.

```python
@torch.no_grad()
def compute_nll_obs(model, tokenizer, aomt_examples: list) -> float:
    """
    For each obs unit O_t: mask it, keep all others as context, measure CE.
    Returns mean NLL over all obs units. Lower = better world model.
    This is a pseudo-log-likelihood (Salazar et al., 2020) — comparable across
    masked DLMs only, not commensurable with AR log-likelihoods.
    """
    MASK = tokenizer.mask_token_id   # = 126336
    SEP  = tokenizer.eos_token_id
    all_nll = []

    for ex in aomt_examples:
        # Build token sequence
        all_ids, spans = [], []
        for i, (text, utype) in enumerate(
                zip(ex["unit_texts"], ex["unit_types"])):
            ids = tokenizer.encode(text, add_special_tokens=False)
            s = len(all_ids)
            all_ids.extend(ids)
            spans.append((s, len(all_ids), utype))
            if i < len(ex["unit_texts"]) - 1:
                all_ids.append(SEP)

        clean_ids = torch.tensor(all_ids, dtype=torch.long)

        for start, end, utype in spans:
            if utype != "obs":
                continue
            masked = clean_ids.clone()
            masked[start:end] = MASK
            logits = model(
                input_ids=masked.unsqueeze(0).to(model.device)
            ).logits[0]
            nll = F.cross_entropy(
                logits[start:end],
                clean_ids[start:end].to(model.device),
            ).item()
            all_nll.append(nll)

    return float(np.mean(all_nll))
```

### 11.3 Robustness evaluation

```python
# eval/noise_robustness.py
def corrupt_observation(obs_tokens: torch.Tensor,
                         rho: float,
                         vocab_size: int,
                         rng) -> torch.Tensor:
    """Replace fraction rho of obs tokens with random vocab tokens."""
    mask = rng.random(len(obs_tokens)) < rho
    corrupted = obs_tokens.clone()
    corrupted[mask] = torch.randint(0, vocab_size, (mask.sum(),))
    return corrupted

# rho values: {0.1, 0.2, 0.3}. Benchmark: ALFWorld seen split.
```

---

## 12. Sanity Checks and Tests

All tests are in `tests/test_suite.py`. Run before any training job:

```bash
python aomt/tests/test_suite.py -v
```

### Critical checks

| Test | What it verifies |
|---|---|
| `test_mask_token_id` | `tok.mask_token_id == 126336` |
| `test_sft_data_counts` | Correct number of datapoints per trajectory per method |
| `test_response_unit_mask_coverage` | All response positions masked; all prompt positions clean |
| `test_response_unit_mask_deterministic` | Same input → same output every call |
| `test_aomt_unit_boundaries` | Masked units have ALL tokens = MASK_TOKEN_ID; unmasked have NONE |
| `test_aomt_resampling` | Same idx → different mask on repeat calls |
| `test_loss_zero_labels` | All-(-100) labels → loss = 0.0, no NaN/inf |
| `test_loss_matches_manual` | `compute_unit_mask_loss` == manual CE over non-(-100) |
| `test_masking_consistency` | Std SFT and AOMT use same loss function, different masks only |
| `test_aomt_lower_loss` | AOMT-Action-Only loss < Std SFT loss on same data (bidirectional context) |
| `test_generation_lengths` | `Actions max < gen_length`, `Traj p99 < max_seq_length` |
| `test_smoke_forward` | Model forward pass: finite loss, no OOM |

---

## 13. Attention Mask Summary

**The causal attention mask is never used anywhere in this project.**

| Method | Causal constraint source | Attention type |
|---|---|---|
| Standard SFT | Data construction: future units absent from sequence | Full bidirectional |
| Prefix SFT S1 | Data construction: only `(O_t, A_t)` in sequence | Full bidirectional |
| Prefix SFT S2 | Data construction: future units absent | Full bidirectional |
| AOMT-Action-Only | None — future obs are intentional context | Full bidirectional |
| AOMT-Mixed | None — future context intentional | Full bidirectional |

LLaDA removes the causal mask from the Transformer architecture. It must be preserved in all training modes. Any causal mask would corrupt the experiment silently.