# AOMT Repository Audit

## 1. File Descriptions

| File Path | Description |
|-----------|-------------|
| `aomt/train.py` | Training entry point CLI. |
| `aomt/evaluate.py` | Evaluation entry point CLI. |
| `aomt/ablate.py` | Ablation study runner CLI. |
| `aomt/data/tokenize_trajectory.py` | Trajectory tokenization and UnitSpan extraction. |
| `aomt/data/masking.py` | Pure masking functions (unit-level Bernoulli). |
| `aomt/data/dataset.py` | Unified AOMTDataset supporting 5 methods. |
| `aomt/data/collator.py` | Padding-aware batch collation. |
| `aomt/model/llada_wrapper.py` | Model/tokenizer loader for LLaDA 2.0-mini. |
| `aomt/model/inference.py` | Mode A (Myopic) and Mode B (Planning) inference loops. |
| `aomt/training/trainer.py` | Unified trainer using HF Accelerator. |
| `aomt/training/losses.py` | Masked cross-entropy loss implementation. |
| `aomt/evaluation/eval_alfworld.py` | ALFWorld environment agent loop. |
| `aomt/evaluation/metrics.py` | NLL_obs and other performance metrics. |
| `aomt/config/base.yaml` | Global hyperparameters. |
| `aomt/slurm/run_all.sh` | Legacy/partial master submission script. |
| `scripts/download_data.py` | ETO dataset download and stats script. |
| `scripts/setup_all.sh` | Main environment setup script. |
| `cluster_info.md` | Cluster hardware and Slurm documentation. |

## 2. Wrong, Duplicated, or Overcomplicated

- **Redundant Scripts:** `scripts/slurm/` contains many individual scripts that overlap with `aomt/slurm/`.
- **Config Duplication:** `aomt/configs/` and `aomt/config/` both exist; `aomt/config/` is the newer version.
- **Inference Redundancy:** `aomt/inference.py` and `aomt/model/inference.py` both exist.
- **Eval Fragmentation:** `aomt/eval/` and `aomt/evaluation/` both exist.
- **Setup Duplication:** `aomt/setup.sh`, `aomt/full_setup.sh`, and `scripts/setup_all.sh` are redundant.

## 3. Implementation Plan

| Action | Files / Directories |
|--------|---------------------|
| **KEEP** | `aomt/data/`, `aomt/model/`, `aomt/training/`, `aomt/evaluation/`, `aomt/config/`, `aomt/weights/`, `requirements.txt`. |
| **ARCHIVE** | `aomt/configs/`, `aomt/eval/`, `aomt/inference.py`, `aomt/run_full_eval.py`, `scripts/slurm/`. |
| **DELETE** | `data/cache/` (will be regenerated), redundant top-level `.py` files in `aomt/`. |

## 4. Full Cluster Configuration Table

| Parameter | Value (Verbatim) |
|-----------|-------|
| Scheduler | Slurm |
| Partitions | `normal`, `gpu-long` |
| GPU Types | V100, Titan V, Titan RTX, T4, A100 (40/80GB), H100 (47/96GB), H200 (141GB) |
| Max Walltime | 20:00:00 (gpu-long) |
| RAM per Node | 32GB to 1TB |
| GPUs per Node | 1 to 8 |
| Module Load | `module load python/3.11` |
| $SCRATCH path | **CLUSTER_TODO** |
| Account | **CLUSTER_TODO** |
| Internet Access | **CLUSTER_TODO** |
| Special Flags | `-G <gpu_type>`, `-C <cuda_version>` |

## 5. dFactory Status

- **Status:** Present at `aomt/dFactory`.
- **URL:** `https://github.com/inclusionAI/dFactory.git` (Note: Prompt suggests ML-GSAI repo, will re-clone if mismatch confirmed).

## 6. Phase Completion Status

- [x] AUDIT.md (Updated)
- [ ] dFactory freshly cloned (Needs verification vs ML-GSAI repo)
- [x] requirements.txt exists
- [x] data/tokenize_trajectory.py exists
- [x] data/masking.py exists
- [x] data/dataset.py exists
- [x] model/llada_wrapper.py exists
- [x] model/inference.py exists (Contains both Mode A and Mode B)
- [x] training/trainer.py exists
- [ ] evaluation scripts for all three benchmarks (AlfWorld exists, others partial)
- [ ] 2 SLURM master scripts exist (Partial/Redundant versions only)

## 7. CLUSTER_TODO

- Confirm $SCRATCH path.
- Identify the correct `#SBATCH --account` for the allocation.
- Verify if compute nodes have internet access for HuggingFace.
- Re-clone dFactory from `https://github.com/ML-GSAI/dFactory` to ensure latest version.
