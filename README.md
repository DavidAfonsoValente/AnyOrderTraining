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

## 3. Training Methods

1.  **Standard SFT:** Baseline causal-prefix SFT.
2.  **Prefix SFT Stage 1:** Offline Internal World Model (IWM) pretraining (local 3-unit context).
3.  **Prefix SFT Stage 2:** Policy SFT initialized from Stage 1.
4.  **AOMT-Mixed:** The proposed method. Joint trajectory modeling via random unit-level masking.

## 4. Inference (Unified)

All methods use **identical inference** to ensure a fair comparison of training objectives.
-   **Method:** LLaDA Block Diffusion (32 steps).
-   **Prompt:** Full trajectory history joined with `\n`, wrapped in a single `USER` turn.
-   **Target:** Next action generated in the `ASSISTANT` turn.

## 5. Design Decisions

- **Unit-Level Masking:** We mask entire text spans (units) to prevent intra-unit information leakage.
- **Data-Driven Causality:** No causal attention mask is used; causality is enforced by sequence construction.
- **Representational AOMT:** Benefit of AOMT-Mixed is internal; no hallucinated planning is used at inference time.

## 6. Testing

Run local (CPU-only) tests:
```bash
pytest aomt/data/tests/ aomt/tests/ -v -m "not gpu"
```

## 7. Documentation

- `PROJECT.md`: Comprehensive technical and methodological documentation.
- `QA_REPORT.md`: Detailed audit findings and bug fixes.
