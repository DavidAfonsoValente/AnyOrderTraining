# AOMT: Any-Order Masked Training for LLM Agents

This repository provides a paper-ready implementation of Any-Order Masked Training (AOMT) for LLM-based agents, optimized for the SoC Compute Cluster (Slurm).

## Overview

AOMT jointly models the entire future trajectory by masking units (observations and actions) at random. This approach allows the agent to act as both a policy and a world model, enabling multi-step planning during inference.

## Quick Start (Slurm)

1.  **Launch Training:**
    ```bash
    bash aomt/slurm/run_train.sh
    ```
2.  **Launch Evaluation:**
    ```bash
    bash aomt/slurm/run_eval_and_analysis.sh
    ```

## Setup

```bash
bash scripts/setup_all.sh
```

## Running Tests

All tests must pass before submitting jobs:
```bash
pytest aomt/data/tests/ aomt/model/tests/ aomt/tests/ -v
```

## Inference Modes

- **Mode A (Myopic):** Predicts only the immediate next action $A_t$. Valid for all training methods.
- **Mode B (Planning):** Uses a multi-step future template (masking $A_t, O_{t+1}, A_{t+1}, \dots$). The model jointly denoises the entire suffix. Only the first action $A_t$ is executed. Exclusive to `aomt_mixed`.
  - **Numerical Invariant:** When planning horizon $H=1$, Mode B is numerically identical to Mode A.

## Design Decisions

- **Unit-Level Masking:** We mask entire units (complete text spans) rather than individual tokens to prevent intra-unit information leakage.
- **Data-Driven Causality:** No causal attention mask is used. Causality is enforced by simply excluding future units from the sequence in SFT methods.
- **Local World Model:** Prefix SFT Stage 1 uses a local 3-unit window ($O_t, A_t, O_{t+1}$) to replicate the ALEE formulation.
- **Mask Resampling:** For AOMT, masks are resampled every epoch in the `__getitem__` call, ensuring the model sees different masking patterns for the same trajectory over time.

## File Structure

```
aomt/
├── data/          # Tokenization, masking, dataset
├── model/         # LLaDA wrapper, inference loops
├── training/      # Unified trainer, loss functions
├── evaluation/    # ALFWorld, ScienceWorld, WebShop runners
├── config/        # YAML hyperparameters
├── slurm/         # Master and individual submission scripts
└── analysis/      # Table and plot generation
```
