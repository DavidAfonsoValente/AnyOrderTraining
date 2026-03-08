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
# Initialize conda for the current session and activate
source ~/miniconda3/etc/profile.d/conda.sh
conda activate py311 && source activate_env.sh
```

### 2. Preparation & Verification (Compute Node)
*Heavy lifting: merging MoE weights, processing trajectories, and running the verification suite. LLaDA 2.0 Mini (MoE) is a **16B parameter model** requiring >32GB VRAM for full weight loading in bf16.*
```bash
# Request an interactive compute node (A100 40GB recommended to fit the 16B model)
salloc --time=2:00:00 --mem=128G --cpus-per-task=8 --gres=gpu:a100-40:1
```

Once the allocation is granted:
```bash
cd ~/AnyOrderTraining/aomt
source activate_env.sh

# A. Prepare model weights (merged-expert format)
srun ./scripts/prepare_model.sh

# B. Download and process trajectories
srun ./scripts/01_prepare_data.sh

# C. Run verification suite (Unit tests + integration)
srun scripts/run_tests.sh

# D. Visualize masking (Sanity check configs)
srun scripts/visualize_experiments.sh

exit # Return to login node
```

### 3. Launch Full Pipeline (Login Node)
*Submits all training variants and evaluation to Slurm. Training scripts (`02-06`) target **multi-GPU H100** by default for maximum performance.*
```bash
source activate_env.sh
bash scripts/submit_pipeline.sh --email your_email@comp.nus.edu.sg
```

---

## 🖥️ Cluster GPU Guide

| Purpose | Recommended GPU | Slurm Flag |
| :--- | :--- | :--- |
| **Interactive Debug** | A100 (40GB) | `--gres=gpu:a100-40:1` |
| **Verification** | A100 (40GB) | `--gres=gpu:a100-40:1` |
| **Training (FSDP)** | 4x A100 (40GB) | `--gpus-per-node=a100-40:4` |
| **High-Perf (FSDP)** | 4x H100 (96GB) | `--gpus-per-node=h100-96:4` |

*Note: The LLaDA 2.0 Mini MoE model (16B) will OOM on Titan RTX (24GB) and Tesla T4 (16GB). Always use at least an A100 (40GB) for single-GPU tasks.*

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
*   `training/`: Legacy components and utility implementations.
*   `eval/`: Full evaluation suite (ALFWorld, ScienceWorld, WebShop, NLL-obs).
*   `scripts/`: Slurm batch scripts and utility scripts (ordered `01` to `07`).
*   `configs/`: YAML configuration files for all experiment modes.

---

## 📖 Documentation
*   `../project_desc.md`: Research proposal and theoretical background.
*   `../engineering_specs.md`: Detailed implementation guide and verification protocols.
*   `../cluster_info.md`: Slurm guide for the SoC Compute Cluster.
