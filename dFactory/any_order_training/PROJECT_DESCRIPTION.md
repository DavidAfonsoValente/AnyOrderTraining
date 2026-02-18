
# Any-Order Masked Training for Trajectory-Level Learning in LLM-Based Agents

## 1. Project Summary

This project implements **Any-Order Masked Training**, a supervised fine-tuning (SFT) paradigm for LLM-based agents. It treats agent learning as a trajectory-level reconstruction problem. Instead of a fixed directional prediction order (e.g., next-action prediction), any-order training repeatedly samples arbitrary masks at the observation/action level across a trajectory. The model is then trained to reconstruct the masked elements from the remaining bidirectional context in a single forward pass.

This approach subsumes standard SFT and prefix-based early-experience methods as special cases, while remaining entirely within the reward-free SFT framework. The implementation is built upon the LLaDA 2.0 model and the dFactory training framework.

## 2. Methodology

### 2.1. The "Order" in Trajectory Learning

In traditional sequence modeling for agents, an "order" is the fixed directional assumption that divides a trajectory into a given context and a target to be predicted.
- **Causal SFT**: `Obs_0, Act_0, ..., Obs_t -> [MASK:Act_t]`
- **Prefix-based (Early-Experience)**: `Obs_0, Act_0 -> [MASK:Obs_{1..T}]`

These single-order methods are effective but limit the model's ability to learn more flexible conditional dependencies within a trajectory.

### 2.2. Any-Order Training

Any-order training removes this constraint by framing learning as a trajectory-level reconstruction task. For each training step, any subset of observations or actions can be masked and predicted from the unmasked elements.

**Key Mechanics:**
- **Masking Granularity**: Masking is applied at the level of entire observations or actions, not individual tokens.
- **Single Forward Pass**: The model receives the unmasked elements and is trained to reconstruct the masked ones in a single pass.
- **Multi-Epoch Resampling**: The mask is resampled for each trajectory across different epochs, exposing the model to a wide variety of context/target configurations over time.

This process allows the model to learn a rich family of conditional distributions `p(masked | unmasked)` without the need for complex iterative diffusion schedules.

## 3. Implementation Details

The project is implemented as a new module within the `dFactory` framework.

### 3.1. Project Structure

The code is organized in the `any_order_training/` directory:
- `configs/`: YAML configuration files for experiments.
- `data/`: Data generation and transformation scripts.
- `sampler/`: The `AnyOrderMaskSampler` implementation.
- `slurm/`: SLURM batch scripts for cluster execution.
- `tasks/`: The main training task script, `train_any_order.py`.
- `tests/`: Unit tests for the custom components.

### 3.2. Core Components

- **`AnyOrderMaskSampler`**: A flexible sampler that masks observation/action units in a trajectory with a given probability `p`. It provides modes for standard SFT (`causal`) and prefix-based (`prefix`) training to enable direct comparisons.

- **Data Pipeline**:
  1.  **Trajectory Generation**: The `data/generate_trajectories.py` script uses the expert bot in `minigrid` (with BabyAI environments) to generate expert trajectories. Observations (grid-based) and actions (discrete) are serialized into textual format.
  2.  **Data Transformation**: The `data/transform.py` script contains the `process_any_order_sft_example` function. This function takes a trajectory, applies the `AnyOrderMaskSampler` according to the selected `training_mode`, and tokenizes the result into `input_ids` and `labels` for the model.

- **Training Task**: The `tasks/train_any_order.py` script, adapted from `dFactory`'s original training script, orchestrates the training process. It uses the custom data pipeline and is configurable via the YAML files in `configs/`.

## 4. Experimental Plan

The project is divided into three phases.

### Phase 1: Implementation and Setup (Complete)

This phase involved implementing all the core components described above, setting up the project structure, and preparing for experiments.

### Phase 2: Small-Scale Ablations (MiniGrid/BabyAI)

This phase will focus on small-scale experiments to validate the approach and tune hyperparameters. The necessary configuration files and evaluation scripts have been prepared.

- **Experiment Configurations**: The `configs/` directory contains YAML files for each experiment:
  - `causal_sft.yaml`: For the causal SFT baseline (`training_mode: causal`).
  - `prefix_sft.yaml`: For the prefix-based SFT baseline (`training_mode: prefix`).
  - `any_order_p15.yaml`: Any-order training with `mask_prob: 0.15`.
  - `any_order_p30.yaml`: Any-order training with `mask_prob: 0.30`.
  - `any_order_sft.yaml`: Any-order training with `mask_prob: 0.50`.

- **Baselines**:
  - **Causal SFT**: `training_mode: causal`
  - **Prefix-based**: `training_mode: prefix`

- **Ablation Studies**:
  - **Mask Probability `p`**: Sweeping `p` from 15% to 50%.
  - **Masking Ratios**: Investigating the effect of masking only observations vs. only actions (can be done by modifying the `process_any_order_sft_example` function).

- **Evaluation**:
  - **World Model NLL**: The `scripts/evaluate.py` script is implemented to calculate the Negative Log-Likelihood of a trained model on a test set. This will be the primary metric for evaluating the world model's quality.
  - **Task Success Rate**: The evaluation script includes a placeholder for this metric. Its implementation requires an interactive environment with the agent, which is pending the setup of the `gymnasium` and `minigrid` libraries.

### Phase 3: Large-Scale Benchmarks (WebArena/ToolBench)

Based on the results from Phase 2, this phase will scale up the experiments to more complex and realistic benchmarks.

- **Environments**: WebArena and ToolBench.
- **Evaluation**: Comparing the best any-order model from Phase 2 against the baselines on these richer benchmarks. The focus will be on task success, robustness to noise, and out-of-distribution generalization.

## 5. Timeline

- **Week 1–2 (Completed)**: Implemented the any-order mask sampler and integrated it into the dFactory training framework.
- **Week 3–6**: Conduct small-scale ablations and world-model evaluations in MiniGrid/BabyAI.
- **Week 7–14**: Run full benchmarks on WebArena/ToolBench, analyze results, and prepare a report or paper draft.

## 6. Deliverables

- A code branch implementing any-order mask sampling and the masked observation/action reconstruction objective in dFactory for LLaDA 2.0.
- Ablation study results showing effects of mask probability, action-vs-observation masking ratios, and comparisons with SFT/prefix baselines.
- Full benchmark results on selected environments with analysis of world-model NLL, task success, and robustness under noise and OOD.
- A final report or paper draft summarizing the findings.
