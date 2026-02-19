# Any-Order Masked Training for Trajectory-Level Learning

## 1. Project Summary

This project implements and evaluates **Any-Order Masked Training**, a supervised fine-tuning (SFT) paradigm for LLM-based agents. It treats agent learning as a trajectory-level reconstruction problem. Instead of a fixed directional prediction order (e.g., next-action prediction), any-order training repeatedly samples arbitrary masks at the observation/action level across a trajectory. The model is then trained to reconstruct the masked elements from the remaining bidirectional context in a single forward pass.

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

### 3.1. Core Components
- **Expert Factory (`generate_trajectories.py`)**: A self-contained script that generates perfect, expert trajectories for a set of canonical `minigrid` tasks. It contains simple, rule-based bots for each task, removing the dependency on the broken, external `babyai` library.
- **Any-Order Sampler (`sampler/`)**: A flexible sampler that implements the core masking logic. It supports ablations on both the *probability* of masking (`mask_prob`) and the *strategy* of masking (`masking_strategy`: all, observations, or actions).
- **Data Pipeline (`transform.py`)**: A data transform that integrates the sampler into the training process, preparing masked trajectories for the model.
- **Training Task (`train_any_order.py`)**: The main training script, which is configurable via YAML files to run any of the planned experiments.

### 3.2. Automation Scripts
- **`setup_and_test.sh`**: A one-time SLURM batch script that automates the entire project setup, including environment creation, dependency installation, model download, data generation, and a final smoke test to verify the pipeline.
- **`run_ablations.sh`**: A simple script to launch the full suite of Phase 2 experiments on a SLURM cluster.

## 4. Phase 2: Comprehensive Ablation Study

This phase is designed to rigorously test the "Any-Order" hypothesis on a controlled set of tasks.

### 4.1. Tasks: A Gradient of Complexity
We use three `minigrid` environments to test the methods against tasks of increasing complexity:
1.  **`minigrid-GoToDoor-v0`**: Simple navigation.
2.  **`minigrid-PickupDist-v0`**: Navigation + simple interaction.
3.  **`minigrid-Unlock-v0`**: Navigation + sequential tool-use.

### 4.2. Ablation Study Design
For each of the three tasks, we will run a full suite of experiments:
- **Baselines**:
  - **Causal SFT**: `training_mode: causal`
  - **Prefix-based**: `training_mode: prefix`
- **Ablation 1: Masking Probability**:
  - `any_order` with `mask_prob` of 0.15, 0.30, 0.50, and 0.75.
- **Ablation 2: Masking Strategy**:
  - `any_order` (with `p=0.50`) masking `observations_only`.
  - `any_order` (with `p=0.50`) masking `actions_only`.

This results in a comprehensive set of ~30 experiments. All configuration files are provided in the `any_order_training/configs/phase2/` directory.

### 4.3. Evaluation Metric
- **Primary Metric**: **World Model NLL (Negative Log-Likelihood)**. Calculated on a held-out test set for each task using the `scripts/evaluate.py` script. A lower NLL indicates a better-learned world model, as the model is less "surprised" by the ground-truth trajectories.

## 5. Timeline & Next Steps
- **Phase 1 (Complete)**: Core logic implementation and setup.
- **Phase 2 (Ready to Run)**: Comprehensive ablation study on controlled data. The project is now fully implemented and ready for this phase to be executed.
- **Phase 3 (Future Work)**: Based on the results from Phase 2, scale up the experiments to more complex and realistic benchmarks like WebArena and ToolBench.
