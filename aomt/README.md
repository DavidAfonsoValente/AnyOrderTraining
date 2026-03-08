# Any-Order Masked Training (AOMT)

This repository contains the implementation for **"Any-Order Masked Training for Trajectory-Level Supervised Learning in LLM-Based Agents,"** a supervised fine-tuning (SFT) paradigm that reframes agent learning as a trajectory-level reconstruction problem.

AOMT uses a masked diffusion language model (LLaDA 2.0) to reconstruct arbitrarily masked observation and action units from bidirectional context in a single forward pass.

---

## 🚀 Ideal Workflow (SoC Compute Cluster)

Perform these steps in order from the `aomt/` directory.

### 1. Environment Setup (Login Node)
*One-time setup. Installs Miniconda, creates a py311 environment, and sets up dependencies.*
```bash
./full_setup.sh
conda activate py311 && source activate_env.sh
```

### 2. Model & Data Preparation (Compute Node)
*Heavy lifting: merging model weights (MoE) and processing trajectories. Requires high memory (~128G).*
```bash
# Request an interactive compute node
salloc --time=2:00:00 --mem=128G --cpus-per-task=8 --gpus=1 --ntasks=1

# Once on the compute node:
cd ~/AnyOrderTraining/aomt # Adjust path if necessary
source activate_env.sh

# A. Prepare model weights (converts to merged-expert format)
srun ./scripts/prepare_model.sh

# B. Download and process trajectories
srun ./scripts/01_prepare_data.sh

exit # Return to login node
```

### 3. Verification (Login Node)
*Ensure masking and core logic are correct before submitting long jobs.*
```bash
source activate_env.sh
bash scripts/run_tests.sh
bash scripts/visualize_experiments.sh
```

### 4. Launch Experiments (Login Node)
*Submits all training variants and evaluation to Slurm with correct dependencies.*
```bash
bash scripts/submit_pipeline.sh --email your_email@comp.nus.edu.sg
```

---

## 📊 Monitoring & Management

*   **Check Queue:** `squeue -u $USER`
*   **Check Job Status:** `sacct -j <jobid>`
*   **Cancel Jobs:** `scancel <jobid>`
*   **View Results:** Collated results appear in `results/summary_table.txt` after the evaluation job completes.

---

## 📂 Repository Structure

*   `data/`: Trajectory parsing, downloading, and mode-specific JSONL generation.
*   `tasks/`: Training entry points for standard SFT and AOMT modes.
*   `training/`: Core implementation of unit-level masking and datasets.
*   `eval/`: Full evaluation suite (ALFWorld, ScienceWorld, WebShop, NLL-obs).
*   `scripts/`: Slurm batch scripts and utility scripts (ordered `01` to `07`).
*   `configs/`: YAML configuration files for all experiment modes.

---

## 📖 Documentation
*   `../project_desc.md`: Research proposal and theoretical background.
*   `../engineering_specs.md`: Detailed implementation guide and verification protocols.
*   `../cluster_info.md`: Slurm guide for the SoC Compute Cluster.
