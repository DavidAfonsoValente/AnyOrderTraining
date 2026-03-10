# Any-Order Masked Training (AOMT)

This repository contains the implementation for **"Any-Order Masked Training for Trajectory-Level Supervised Learning in LLM-Based Agents,"** a supervised fine-tuning (SFT) paradigm that reframes agent learning as a trajectory-level reconstruction problem.

AOMT uses a masked diffusion language model (LLaDA 2.0) to reconstruct arbitrarily masked observation and action units from bidirectional context in a single forward pass.

---

## 🚀 Automated Workflow (SoC Compute Cluster)

Perform these steps from the `aomt/` directory.

### 1. Environment Setup (Compute Node via Slurm)
*One-time setup. Installs Miniconda, creates a py311 environment, and sets up all dependencies.*

**⚠️ DO NOT run this on a login node.** It will fail with `CondaMemoryError`.

```bash
sbatch slurm_setup.sh
# Monitor with squeue -u $USER
```

### 2. Preparation (Native GPU Compute Node)
*Heavy lifting: merging MoE weights and processing trajectories.*

**⚠️ Native GPU Required:** Do NOT use MIG (partitioned) GPUs for weight merging.

```bash
# Request a Native H100 or A100-80GB node
salloc --time=1:00:00 --mem=128G --cpus-per-task=8 --gpus-per-node=h100-96:1
```

Once allocated:
```bash
source activate_env.sh
bash scripts/prepare_model.sh  # Merges MoE experts for dFactory
bash scripts/01_prepare_data.sh # Pre-processes trajectories
exit
```

### 3. Launch Full Automated Pipeline (Login Node)
*Submits the end-to-end self-tuning pipeline to Slurm.*

```bash
source activate_env.sh
bash scripts/submit_pipeline.sh --email your_email@comp.nus.edu.sg
```

---

## 🤖 How the Pipeline Works (Self-Tuning)

The pipeline is fully automated and operates in three distinct phases to ensure optimal results and QOS compliance:

### Phase 1: Hyperparameter Search (Ablation)
The system automatically runs a 4-way sweep of `mask_prob` ∈ {0.15, 0.25, 0.40, 0.50} on a representative ALFWorld subset. These runs are chained sequentially to stay within cluster GPU quotas.

### Phase 2: Automatic Selection (The "Judge")
Once the sweep finishes, a specialized job (`aomt_update_config`) automatically:
1. Scans all ablation result files.
2. Identifies the probability with the highest success rate.
3. **Physically updates the YAML configuration files** for the main experiments with the winning value.

### Phase 3: Main Benchmarking
The primary experiments (Standard SFT, Prefix S1/S2, AOMT-Action, AOMT-Mixed) execute sequentially using the optimal hyperparameter. The pipeline concludes by running the full evaluation suite and generating a summary table.

---

## 🖥️ Cluster Resource Strategy

| Feature | Implementation | Rationale |
| :--- | :--- | :--- |
| **QOS Compliance** | **Sequential Chaining** | Only one 2-GPU task is active at a time to satisfy strict cluster quotas. |
| **Hardware** | **1x H100-96 Node** | Provides exactly 192GB VRAM—the minimum required for the 16B model. |
| **Memory** | **Grad Checkpointing** | Enabled by default to fit the 16B MoE sharded across 2 GPUs. |
| **Stability** | **Pre-flight Checks** | Downstream jobs verify checkpoints before starting intensive loops. |

---

## 📊 Monitoring & Management

*   **Check Queue:** `squeue -u $USER`
*   **Check Job Status:** `sacct -j <jobid>`
*   **Cancel Pipeline:** `scancel -u $USER`
*   **View Results:** 
    *   Ablation results: `results/ablation_p*_alfworld.json`
    *   Final comparison: `results/summary_table.txt`

---

## 📂 Repository Structure

*   `data/`: Trajectory parsing, subset filtering, and JSONL generation.
*   `tasks/`: Training entry points for standard SFT and AOMT modes.
*   `eval/`: Full evaluation suite (ALFWorld, ScienceWorld, WebShop, NLL-obs).
*   `scripts/`: Automation logic (`submit_pipeline.sh`, `apply_ablation_winner.sh`).
*   `configs/`: Master YAML files (automatically tuned by the pipeline).
