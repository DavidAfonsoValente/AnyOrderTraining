# Any-Order Masked Training (AOMT)

This repository contains the implementation for **"Any-Order Masked Training for Trajectory-Level Supervised Learning in LLM-Based Agents,"** a supervised fine-tuning (SFT) paradigm that reframes agent learning as a trajectory-level reconstruction problem.

AOMT uses a masked diffusion language model (LLaDA 2.0) to reconstruct arbitrarily masked observation and action units from bidirectional context in a single forward pass.

---

## 1. Quick Start (SoC Compute Cluster)

All commands should be run from the `aomt/` directory on a Slurm login node (`xlogin0`, `xlogin1`, or `xlogin2`).

### Step 1: Full Environment Setup
This script installs Miniconda (if needed), creates a Python 3.11 environment, clones `dFactory`, and sets up the project virtual environment.
```bash
./full_setup.sh
```
**After setup finishes, you MUST activate the environment in your current shell:**
```bash
conda activate py311
source activate_env.sh
```

### Step 2: Model Preparation
Prepare the `LLaDA2.0-mini` weights for training (converts to merged-expert format).
```bash
./scripts/prepare_model.sh
```

### Step 3: Launch the Full Pipeline
Submit the entire data preparation, training (all 5 variants), and evaluation suite to Slurm with correct job dependencies.
```bash
bash scripts/submit_pipeline.sh
```
*Note: You can optionally provide your email for Slurm notifications:* `bash scripts/submit_pipeline.sh --email user@comp.nus.edu.sg`

---

## 2. Detailed Commands

### Environment Management
*   **Activate Project Env:** `source activate_env.sh` (run this in every new terminal session).
*   **Run Unit Tests:** `bash scripts/run_tests.sh`
*   **Run Sanity Checks:** `python scripts/run_sanity_checks.py`
*   **Visualize Masking:** `bash scripts/visualize_experiments.sh`

### Data Preparation
If you need to run data preparation manually (e.g., for debugging):
1.  **Request Compute Node:** `salloc --time=2:00:00 --mem=64G --cpus-per-task=8`
2.  **Run Prep Script:** `srun scripts/01_prepare_data.sh`

### Individual Training Jobs
To submit a specific training variant manually:
*   **Standard SFT:** `sbatch scripts/02_train_sft_standard.sh`
*   **Prefix SFT S1:** `sbatch scripts/03_train_prefix_s1.sh`
*   **Prefix SFT S2:** `sbatch scripts/04_train_prefix_s2.sh` (requires S1 checkpoint)
*   **AOMT-Action-Only:** `sbatch scripts/05_train_aomt_action.sh`
*   **AOMT-Mixed:** `sbatch scripts/06_train_aomt_mixed.sh`

### Monitoring and Management
*   **Check Queue:** `squeue -u $USER`
*   **Check Job Status:** `sacct -j <jobid>`
*   **Cancel Job:** `scancel <jobid>`
*   **View Logs:** Logs are saved in `aomt/logs/` and `slurm-<jobid>.out` in the submission directory.

---

## 3. Repository Structure

*   `data/`: Scripts for downloading, processing, and analyzing trajectories.
*   `tasks/`: Custom training task entry points for dFactory.
*   `training/`: Core implementation of unit-level masking and datasets.
*   `eval/`: Full evaluation suite (ALFWorld, ScienceWorld, WebShop, NLL-obs).
*   `scripts/`: Slurm batch scripts and utility scripts.
*   `configs/`: YAML configuration files for all experiment modes.

---

## 4. Technical Documentation
*   `../project_desc.md`: Research proposal and theoretical background.
*   `../engineering_specs.md`: Detailed implementation guide and verification protocols.
*   `../cluster_info.md`: Slurm guide for the SoC Compute Cluster.
