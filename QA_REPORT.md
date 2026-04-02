# AOMT QA Report

## 1. Repository File Inventory

| File Path | Description | Status |
|-----------|-------------|--------|
| `aomt/train.py` | Training entry point CLI. | OK |
| `aomt/evaluate.py` | Evaluation entry point CLI. | OK |
| `aomt/ablate.py` | Ablation study runner CLI. | NEEDS_FIX (AOA refs) |
| `aomt/data/tokenize_trajectory.py` | Trajectory tokenization and UnitSpan extraction. | OK |
| `aomt/data/masking.py` | Pure masking functions. | OK |
| `aomt/data/dataset.py` | Unified AOMTDataset supporting 5 methods. | NEEDS_FIX (AOA refs) |
| `aomt/model/inference.py` | Mode A and Mode B inference loops. | NEEDS_FIX (Inconsistent SEP) |
| `aomt/training/trainer.py` | Unified trainer using HF Accelerator. | OK |
| `aomt/slurm/run_train.sh` | Master training script. | NEEDS_FIX (AOA refs) |
| `aomt/slurm/run_eval_and_analysis.sh` | Master eval script. | OK |

## 2. Cluster Configuration

| Parameter | Value (Verbatim) |
|-----------|-------|
| Scheduler | Slurm |
| Partitions | `normal`, `gpu-long` |
| GPU Types | V100, Titan V, Titan RTX, T4, A100 (40/80GB), H100 (47/96GB), H200 (141GB) |
| Max Walltime | 20:00:00 (gpu-long), 01:00:00 (normal) |
| RAM per Node | 32GB to 1TB |
| GPUs per Node | 1 to 8 |
| Module Load | `module load python/3.11` |
| $SCRATCH path | **CLUSTER_TODO** |
| Account | **CLUSTER_TODO** |
| Internet Access | **CLUSTER_TODO** |

## 3. dFactory Status

- **URL:** `https://github.com/inclusionAI/dFactory.git`
- **Base Dataset:** `veomni.data.dataset.MappingDataset`
- **Collator:** `veomni.data.data_collator.DataCollatorWithPadding`
- **Provided:** MoE loading, distributed utilities.
- **Implemented here:** Unit masking, trajectory tokenization.

## 4. ETO Dataset Fields

- **Fields:** `id`, `game_file`, `conversations`.
- **Splits:** `webshop`, `scienceworld`, `alfworld` (Sub-splits managed by robust loader).
- **Sample:** `{'from': 'human', 'value': '...'}`.

## 5. LLaDA Tokenizer Facts

- **mask_token_id:** 156895
- **pad_token_id:** 156892
- **vocab_size:** 156891
- **sep token (\n) encoding:** [198]
- **generate() exists?** No (using custom block diffusion loop).

## 6. Bugs Found (Critical)

1.  **Inconsistent SEP Tokenization:** `inference.py:tokenize_history` appends a trailing `SEP` to the prompt, but `tokenize_trajectory.py` only places `SEP` between units. This creates a train-inference mismatch.
2.  **AOA Persistence:** `aomt_action_only` is still present in `train.py`, `dataset.py`, `masking.py`, and `slurm/`. It must be archived.

## 7. Bugs Found (Non-Critical)

1.  **Duplicate Configs:** `aomt/configs/` still exists (should be archived).
2.  **Legacy Scripts:** `scripts/slurm/` and `aomt/setup.sh` are redundant.

## 8. aomt_action_only References

- `aomt/data/dataset.py`: lines 23, 138, 139
- `aomt/data/masking.py`: line 46
- `aomt/ablate.py`: line 10
- `aomt/slurm/run_train.sh`: multiple lines
- `aomt/train.py`: line 14

## 9. Missing Implementations

- All `ablate.py` runners except `run_mask_prob_ablation` are placeholders.
- `analysis/nll.py` logic is a placeholder.

## 10. Phase Completion Status

- Phase 0: **COMPLETE**
- Phase 1: **PARTIAL** (removals pending)
- Phase 2: **PARTIAL** (implementation exists, specs verified)
- Phase 3: **COMPLETE**
- Phase 4: **COMPLETE**
- Phase 5: **COMPLETE**
- Phase 6: **NOT_STARTED** (ablation runners pending)
- Phase 7: **PARTIAL** (analysis entry points exist)
- Phase 8: **COMPLETE** (tests pass)
- Phase 9: **NOT_STARTED** (cluster gpu tests pending)
- Phase 10: **COMPLETE**
- Phase 11: **COMPLETE**
- Phase 12: **COMPLETE**
