# AOMT: Any-Order Masked Training for LLM Agents

## 1. Overview

Any-Order Masked Training (AOMT) is a training framework for LLM-based agents that enables them to jointly model both policies and world models. Unlike standard Supervised Fine-Tuning (SFT), which typically predicts the next action in a causal sequence, AOMT uses a random unit-level masking strategy over the entire agent trajectory. This allows the model to learn bidirectional dependencies between observations and actions, facilitating multi-step planning and increasing robustness to observation noise.

This codebase provides a complete, paper-ready implementation of AOMT, including:
- Trajectory-level unit tokenization and masking.
- Four distinct training methods (Standard SFT, Prefix SFT Stages 1 & 2, and AOMT-Mixed).
- Dual inference modes: Myopic (Mode A) and Planning (Mode B).
- Comprehensive evaluation suite for ALFWorld, ScienceWorld, and WebShop.
- Automated ablation and analysis pipelines.

## 2. Background: Why AOMT?

Standard SFT is inefficient for trajectory modeling. For a trajectory of length $T$, standard SFT only utilizes a small fraction ($T/2^{2T+1}$) of the potential information available in the sequence. By contrast, AOMT's random masking objective allows the model to learn from any combination of context and target units within the trajectory.

We leverage LLaDA 2.0, a masked diffusion language model, which is naturally suited for this task due to its native bidirectional attention and ability to perform iterative denoising (block diffusion).

## 3. The Four Training Methods

### Standard SFT (Method 1)
Baseline causal-prefix policy learning. The model predicts the next action given the history.
**Structure:** `[O0, SEP, A0, SEP, ..., Ot, SEP, MASK_At]`

### Prefix SFT Stage 1: Offline IWM (Method 2)
Pretrains the model as an Internal World Model (IWM) using local context only.
**Structure:** `[Ot, SEP, At, SEP, MASK_Ot+1]` (exactly 3 units).

### Prefix SFT Stage 2: Policy SFT (Method 3)
Fine-tunes the policy starting from the Stage 1 checkpoint.
**Structure:** Same as Standard SFT.

### AOMT-Mixed (Method 4)
The proposed method. Randomly masks any unit (observation or action) in the full trajectory.
**Structure:** `[M_O0, SEP, M_A0, SEP, ..., M_OT]` where $M_i$ is a Bernoulli mask.

## 4. Inference Modes

### Mode A: Myopic Inference
Denoises only the immediate next action slot. Valid for all methods.
1. Tokenize history.
2. Append `[MASK] * max_new_tokens`.
3. Iteratively unmask the suffix using block diffusion.

### Mode B: Planning Inference
Exclusive to `aomt_mixed`. Jointly denoises a multi-step future template.
1. Build planning template: `[History, SEP, MASK_At, SEP, MASK_Ot+1, SEP, MASK_At+1, ...]`
2. Jointly denoise the entire suffix.
3. Extract only $A_t$ for execution; discard the rest.
4. Replan at each step with real observations.

## 5. Design Decisions

- **Unit-Level Masking:** We mask entire text spans (units) to prevent intra-unit information leakage, forcing the model to rely on inter-unit context.
- **Data-Driven Causality:** We enforce causality by simply omitting future units from the sequence in SFT methods, avoiding the need for a causal attention mask.
- **Mask Resampling:** Masks are resampled every epoch during the `__getitem__` call to maximize data diversity.
- **dFactory as a Library:** We use `dFactory/VeOmni` for MoE handling and distributed utilities but keep our AOMT logic separate to prevent dependency divergence.

## 6. Training and Evaluation

Training is orchestrated via two master SLURM scripts:
1. `bash slurm/run_part1_baselines_and_sweeps.sh`: Pre-flight tests and main training runs.
2. `bash slurm/run_part2_aomt_eval_and_analysis.sh`: Evaluation and result generation.

Evaluation is performed on:
- **ALFWorld:** Household task success.
- **ScienceWorld:** Elementary science task scores.
- **WebShop:** E-commerce agent rewards.
- **NLL_obs:** Trajectory reconstruction accuracy (pseudo-log-likelihood).
