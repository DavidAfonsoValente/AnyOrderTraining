# Any-Order Masked Training (AOMT) for Agents

This project implements **Any-Order Masked Training (AOMT)**, a novel fine-tuning paradigm for agentic language models based on **Masked Diffusion Language Models (DLMs)** like LLaDA 2.0.

## Overview

Traditional agent fine-tuning (Standard SFT) typically trains models to predict actions ($A_t$) given a causal history of observations and prior actions ($O_0, A_0, \dots, O_t$). AOMT departs from this by training the model to reconstruct *any* missing part of a trajectory, including observations, from any other part. This treats the entire agent trajectory as a single bidirectional sequence where the model learns a joint policy and implicit world model.

### Key Innovations
1.  **Flat Trajectory Format:** Instead of structured chat templates, trajectories are tokenized as a flat sequence of interleaved observations and actions separated by `[EOS]` tokens.
2.  **Unit-Level Masking:** Training uses unit-level Bernoulli masking (typically $p=0.25$), where entire observations or actions are masked and must be reconstructed from the surrounding context.
3.  **Bidirectional Context:** By masking observations during training, the model learns to understand environment dynamics (Implicit World Model). By masking actions, it learns the policy.
4.  **Training-Inference Alignment:** AOMT is specifically designed to leverage the iterative denoising property of Masked Diffusion models at inference time.

---

## Project Structure

### Core Components (`aomt/`)
- **`data/`**: Scripts for downloading and processing the ETO (Expert Trajectory Optimization) dataset.
  - `prepare_data.py`: Main script to convert ETO trajectories into training-ready JSONL files for SFT, Prefix SFT, and AOMT.
  - `utils.py`: Contains `load_robust_dataset` for handling dataset loading.
- **`training/`**: Implementation of training utilities.
  - `trainer.py`: Custom trainer wrapper.
  - `mask_sampler.py`: Logic for selecting which units to mask during training.
  - `collator.py`: Handles batching and padding.
- **`tasks/`**: High-level training task entry points.
  - `train_aomt.py`: The primary script for AOMT Mixed training.
  - `train_standard_sft.py`: Baseline script for Standard SFT and Prefix SFT.
- **`eval/`**: Evaluation suite for benchmarking models.
  - `task_eval.py`: Core evaluation loop for ALFWorld, supporting both flat and chat inference formats.
  - `compute_nllobs.py`: Computes the Negative Log-Likelihood of observations, used to measure the quality of the implicit world model.
- **`inference.py`**: Implements the iterative denoising logic for both chat-formatted and flat trajectory sequences.

### Infrastructure
- **`scripts/slurm/`**: A complete suite of SLURM scripts for running the full experimental pipeline on a GPU cluster.
  - `pipeline_submit.sh`: Master submission script with job dependencies.
  - `train_aomt_mixed_sweep.sh`: Hyperparameter sweep for masking probability.
  - `eval_steps_ablation.sh`: Verifies performance across different denoising steps (1 to 128).
- **`configs/`**: YAML configuration files defining hyperparameters for each training stage and evaluation setting.

---

## Methodology

### 1. Training Paradigms
- **Standard SFT:** Trained at $p=1.0$ masking on the assistant turn using a chat template. Predicts $A_t$ from causal history.
- **Prefix SFT (Stage 1):** Trains an Implicit World Model by predicting $O_{t+1}$ from the local pair $(O_t, A_t)$.
- **Prefix SFT (Stage 2):** Fine-tunes the Stage 1 model on action prediction (Policy SFT).
- **AOMT-Mixed:** Trains on flat trajectories with Bernoulli unit-level masking. Learns joint policy and world model.

### 2. Inference (Masked Diffusion)
Inference involves starting with a sequence of `[MASK]` tokens for the target unit and iteratively unmasking them based on model confidence.
- **Standard Format:** Uses `tokenizer.apply_chat_template` to generate the next action.
- **Flat Format:** Uses the raw trajectory sequence with `[EOS]` separators, matching AOMT's training distribution.

### 3. Evaluation Metrics
- **Success Rate:** Percentage of tasks successfully completed in environments like ALFWorld.
- **NLLobs:** Measures how well the model predicts observations (world model quality).
- **Noise Robustness ($\rho$):** Evaluates performance when observations are corrupted with noise.

---

## Technical Details

- **Backend:** Powered by `veomni` (from the `dFactory` submodule), which provides optimized FSDP (Fully Sharded Data Parallel) and MoE (Mixture of Experts) support.
- **Model:** Built upon **LLaDA 2.0-mini**, a masked diffusion transformer.
- **Format:** All models are trained on the **Full ReAct string** (`Thought: ... \n Action: ...`) to preserve reasoning capabilities.
