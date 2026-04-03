# AOMT: Any-Order Masked Training for LLM Agents

Complete, paper-ready implementation of **Any-Order Masked Training (AOMT)** for LLM-based agents, optimized for the SoC Compute Cluster (Slurm).

## 1. Quick Start (Cluster)

To run the entire pipeline (tests, training, evaluation, and analysis) on the cluster, execute these two commands:

```bash
# Part 1: Run pre-flight tests and submit all training jobs
bash slurm/run_part1_baselines_and_sweeps.sh

# Part 2: Submit evaluation and analysis (waits for Part 1 to finish)
bash slurm/run_part2_aomt_eval_and_analysis.sh
```

## 2. Environment Setup

The project requires Python 3.10+ and a GPU cluster.

```bash
# 1. Clone the repository
git clone <repo_url>
cd AnyOrderTraining

# 2. Setup environment (installs requirements, clones dFactory, prepares benchmarks)
bash scripts/setup_all.sh
```

## 3. Repository Structure

- `aomt/`: Core implementation.
  - `data/`: Tokenization, unit-level masking, and `AOMTDataset`.
  - `model/`: LLaDA wrapper and Mode A/B inference logic.
  - `training/`: Unified trainer and loss functions.
  - `evaluation/`: Benchmark runners (ALFWorld, ScienceWorld, WebShop).
  - `slurm/`: Individual job scripts and cluster utilities.
- `slurm/`: Master orchestration scripts.
- `scripts/`: Environment setup and data preparation utilities.
- `results/`: Output CSVs, LaTeX tables, and PNG plots (generated at runtime).

## 4. Training Methods

1.  **Standard SFT:** Baseline causal-prefix SFT.
2.  **Prefix SFT Stage 1:** Offline Internal World Model (IWM) pretraining (local 3-unit context).
3.  **Prefix SFT Stage 2:** Policy SFT initialized from Stage 1.
4.  **AOMT-Mixed:** The proposed method. Joint trajectory modeling via random unit-level masking.

## 5. Inference Modes (for AOMT-Mixed)

-   **Mode A (Myopic):** Denoises only the next action slot.
-   **Mode B (Planning):** Jointly denoises a multi-step future template; extracts only the first action.

## 6. Testing

Run local (CPU-only) tests:
```bash
pytest aomt/data/tests/ aomt/tests/ -v -m "not gpu"
```

Run cluster (GPU required) tests:
```bash
sbatch aomt/slurm/run_tests.sh
```

## 7. Documentation

- `PROJECT.md`: Comprehensive methodological and technical documentation.
- `AUDIT.md`: Cluster configuration details and project history.
- `QA_REPORT.md`: Detailed audit findings and bug fixes.
