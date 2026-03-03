# Any-Order Masked Training for Trajectory-Level Learning in LLM-Based Agents

This repository contains the implementation and experimental setup for a research project investigating Any-Order Masked Training (AOMT) for fine-tuning masked Diffusion Language Models (DLMs) as agents. The project aims to compare AOMT against established causal language model training methods (Instruction Learning and Implicit World Modelling) in environments like MiniGrid and ALFWorld.

## Project Structure

*   `dFactory/`: Contains the core LLaDA model implementation, training framework (`VeOmni`), scripts for data generation (MiniGrid), model conversion, and LLaDA-specific SLURM job scripts.
    *   `configs/`: YAML configuration files for LLaDA experiments.
    *   `sampler/any_order_sampler.py`: Implementation of the AOMT, B-SingleUnit, and B-Prefix masking strategies.
    *   `scripts/`: Python scripts for data handling, model evaluation (NLL, rollouts, trajectory completion).
    *   `tasks/train_llada2_bd.py`: The main training script for LLaDA models, integrated with `AnyOrderSampler`.
    *   `transform/trajectory_transform.py`: Utility for processing raw trajectories into tokenized units.
    *   `VeOmni/`: The underlying dFactory framework.
*   `qwen_experiments/`: Contains custom Python scripts and SLURM job scripts for running Qwen-based experiments (Instruction Learning and Implicit World Modelling). This setup is separate due to the architectural differences between LLaDA (masked DLM) and Qwen (causal LM).
*   `models/`: Directory where downloaded LLaDA and Qwen models, as well as converted LLaDA models, are stored.
*   `data/`: Directory for generated MiniGrid datasets and preprocessed ALFWorld expert data.
*   `results/`: Directory where all training checkpoints and evaluation results will be stored.


## Prerequisites

Before running the experiments, ensure you have the following installed:

*   **SLURM Workload Manager:** For job scheduling on a cluster.
*   **Python 3.8+:** The project's Python version.
*   **Poetry:** For Python dependency management. Install it via `pip install poetry` or as per [Poetry's official documentation](https://python-poetry.org/docs/#installation).
*   **CUDA-enabled GPU(s):** For model training and inference.

## Step-by-Step Execution Guide

### 1. Initial Setup and Data Preparation (Phase 1)

This phase sets up the Python virtual environment, downloads all necessary models (LLaDA and Qwen), installs required libraries, and prepares the datasets for both MiniGrid and ALFWorld.

1.  **Run the Phase 1 SLURM script:**
    ```bash
    sbatch dFactory/phase1_prepare_data.slurm
    ```
    This script will:
    *   Execute `dFactory/VeOmni/setup_venv.sh` to create and configure a Poetry virtual environment (`dFactory/VeOmni/.venv`).
    *   Install all Python dependencies, including `alfworld-git` and `rouge_score`.
    *   Download the `LLaDA2.0-mini-preview` model from Hugging Face.
    *   Convert the LLaDA model to the `merged-expert` format required by dFactory.
    *   Download the `Qwen2-1.5B` model from Hugging Face.
    *   Run `alfworld-download --extra` to get the raw ALFWorld data (stored in `~/.cache/alfworld/`).
    *   Preprocess the raw ALFWorld expert data into a JSONL format suitable for Qwen training, saving it to `qwen_experiments/data/alfworld_expert_train.jsonl`.
    *   Generate expert trajectories for the three MiniGrid environments (`GoToDoor-v0`, `PickupDist-v0`, `Unlock-v0`), saving them to `data/minigrid/`.

    **Wait for this job to complete successfully before proceeding.** You can monitor its status with `squeue` or by checking the output logs in `dFactory/logs/`.

### 2. MiniGrid Mask Sampling Ablation (Phase 2)

This phase runs the ablation study on MiniGrid to determine the optimal AOMT configuration ("AOMT-Best") for LLaDA models.

1.  **Run the Phase 2 SLURM job array:**
    ```bash
    sbatch --array=0-71 dFactory/phase2_minigrid_ablation.slurm
    ```
    This will launch 72 individual jobs, covering 8 masking conditions across 3 MiniGrid environments, each with 3 random seeds. Each job will train a LLaDA model and then run placeholder evaluation (which you would need to implement in `dFactory/phase2_minigrid_ablation.slurm` if you want NLL evaluation results directly from this script).

    **Wait for all jobs in this array to complete.**

2.  **Evaluate MiniGrid models (NLLs):**
    The `dFactory/phase2_minigrid_ablation.slurm` script currently has a placeholder for evaluation. To get the NLL metrics as specified, you would need to run the `scripts/evaluate_marginal_nll.py` and `scripts/evaluate_forward_nll.py` scripts on the trained models.

    You can create a separate SLURM script or run them sequentially after all training jobs are complete. For example, to evaluate a single model:
    ```bash
    # Example for one model
    # First, locate the checkpoint for a specific run, e.g., for AO-p50 on Unlock-v0, seed 0
    CHECKPOINT_DIR="/home/davidvalente/AnyOrderTraining/results/phase2/Unlock-v0/AO-p50/seed_0/hf_ckpt"
    DATASET_PATH="/home/davidvalente/AnyOrderTraining/data/minigrid/Unlock-v0_train.jsonl" # Or a dedicated eval set

    python dFactory/scripts/evaluate_marginal_nll.py 
      --model_path "$CHECKPOINT_DIR" 
      --dataset_path "$DATASET_PATH" 
      --output_file "results/phase2/Unlock-v0/AO-p50/seed_0/marginal_nll.json"

    python dFactory/scripts/evaluate_forward_nll.py 
      --model_path "$CHECKPOINT_DIR" 
      --dataset_path "$DATASET_PATH" 
      --output_file "results/phase2/Unlock-v0/AO-p50/seed_0/forward_nll.json"
    ```
    You will need to write a script to automate this for all 72 models.

3.  **Select AOMT-Best:**
    Analyze the `NLL_obs_forward` results, particularly for the `Unlock-v0` environment, from the evaluation data. The AOMT-Best condition is the one with the lowest `NLL_obs_forward` on the Unlock-v0 validation set. If there are near-ties, use `NLL_total` as a tie-breaker. Update the `dFactory/configs/alfworld/LLaDA+AOMT-Best.yaml` file with the `mask_prob` and `mask_strategy` of the selected AOMT-Best configuration.

### 3. ALFWorld Primary Evaluation (Phase 3)

This phase runs the main comparison experiments on ALFWorld, involving both LLaDA and Qwen models.

#### 3.1 LLaDA Conditions Training

1.  **Run LLaDA training job array:**
    ```bash
    sbatch --array=0-8 dFactory/phase3_alfworld_train_llada.slurm
    ```
    This will train the LLaDA+B-SU, LLaDA+B-Pfx, and LLaDA+AOMT-Best conditions (3 seeds each, total 9 jobs) on the ALFWorld dataset.
    **Wait for all jobs in this array to complete.**

#### 3.2 LLaDA Conditions Evaluation

1.  **Run LLaDA evaluation job array:**
    ```bash
    sbatch --array=0-8 dFactory/phase3_alfworld_eval_llada.slurm
    ```
    This will evaluate the 9 trained LLaDA models using rollouts (Task SR), marginal NLL, forward NLL, and trajectory completion.

    **Wait for all jobs in this array to complete.**

#### 3.3 Qwen Conditions Training and Evaluation

1.  **Run Qwen multi-stage job array:**
    ```bash
    sbatch --array=0-5 qwen_experiments/phase3_alfworld_qwen.slurm
    ```
    This will launch 6 jobs, covering Qwen+IL and Qwen+IWM conditions (3 seeds each). This script handles the entire multi-stage process for IWM (IL pre-training, rollout generation, IWM data preparation, and final IWM training) and directly trains for Qwen+IL.

    **Wait for all jobs in this array to complete.**

### 4. Results Analysis

After all jobs are completed, you will find results in the `results/` directory, organized by phase, environment/condition, and seed.

*   **Task Success Rate (SR):** The primary metric for cross-model comparison, found in the output of `run_alfworld_rollout.py` (for both LLaDA and Qwen).
*   **NLL Metrics:** For within-LLaDA comparisons and ablation analysis, found in the output of `evaluate_marginal_nll.py` and `evaluate_forward_nll.py`.
*   **ROUGE-L / Exact Match:** For within-LLaDA trajectory completion quality, found in the output of `trajectory_completion_demo.py`.

Use the generated JSON output files to compile your findings as described in the project specifications.

---

## Important Notes

*   **ALFWorld Data:** The `qwen_experiments/preprocess_alfworld_data.py` script contains placeholder logic for extracting expert trajectories from the raw `alfworld-download` output. You might need to **adapt this script** based on the exact structure of the ALFWorld expert data available in your setup.
*   **Qwen-2.5-7B vs Qwen2-1.5B:** The project specification mentions `Qwen-2.5-7B`, but due to availability and common practices for open-source models, `phase1_prepare_data.slurm` downloads `Qwen/Qwen2-1.5B`. This is a smaller model but suitable for demonstrating the experimental pipeline. If `Qwen-2.5-7B` becomes directly available on Hugging Face, you can update the `repo_id` in `phase1_prepare_data.slurm`.
*   **GPU Memory:** Ensure your cluster nodes have sufficient GPU memory (e.g., 64GB per GPU as specified in SLURM scripts) to handle the models and batch sizes.
*   **Time Limits:** The `--time` limits in the SLURM scripts are estimates. You may need to adjust them based on your cluster's load and the actual runtime of the experiments.
*   **Debugging:** Monitor `dFactory/logs/` and `qwen_experiments/logs/` for job outputs and errors. If a job fails, inspect the `.err` file for details.
*   **Rollout Budget for Qwen+IWM:** The `rollout_qwen_iwm.py` generates 100 episodes for rollouts. The spec mentions using the hyperparameters and rollout budget from Zhang et al. (2025). This might require adjusting `num_episodes` in `qwen_experiments/rollout_qwen_iwm.py` and `qwen_experiments/phase3_alfworld_qwen.slurm`.
*   **`dFactory/phase2_minigrid_ablation.slurm` evaluation:** The script currently trains models but does *not* run the NLL evaluation for Phase 2 results. You would need to add this (as suggested in Section 2, Step 2) or run it manually.

This comprehensive guide should allow you to execute the entire research project pipeline.