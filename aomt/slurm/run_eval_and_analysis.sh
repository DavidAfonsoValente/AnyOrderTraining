#!/bin/bash
# aomt/slurm/run_eval_and_analysis.sh
# Launch all evaluation, ablation-evaluation, and analysis jobs.
# Automatically depends on all training jobs from run_train.sh.
# Usage: bash aomt/slurm/run_eval_and_analysis.sh
set -euo pipefail

echo "════════════════════════════════════════════════════════"
echo "AOMT Evaluation & Analysis Pipeline — Submitting Jobs"
echo "════════════════════════════════════════════════════════"

# Read training job IDs
TRAIN_ID_FILE="aomt/slurm/.train_ids"
if [ ! -f "$TRAIN_ID_FILE" ]; then
  echo "ERROR: $TRAIN_ID_FILE not found. Run bash aomt/slurm/run_train.sh first."
  exit 1
fi
ALL_TRAIN=$(cat "$TRAIN_ID_FILE")
echo "Depending on training jobs: $ALL_TRAIN"
echo ""

# ── Evaluation (all methods, Mode A, all 3 benchmarks) ─────────────────────
J_EVAL_A=$(sbatch --parsable \
           --dependency=afterok:${ALL_TRAIN} \
           aomt/slurm/eval_all_mode_a.sh)

# ── Mode B evaluation (aomt_mixed only, H sweep + best H on all benchmarks) ─
J_EVAL_B=$(sbatch --parsable \
           --dependency=afterok:${ALL_TRAIN} \
           aomt/slurm/eval_aomt_mixed_mode_b.sh)

# ── All ablation evaluations (depends on both training and Mode A eval) ─────
J_ABLATE=$(sbatch --parsable \
           --dependency=afterok:${ALL_TRAIN}:${J_EVAL_A}:${J_EVAL_B} \
           aomt/slurm/ablate_all.sh)

# ── Analysis (tables + figures + NLL) — depends on all eval ─────────────────
J_ANALYSIS=$(sbatch --parsable \
             --dependency=afterok:${J_EVAL_A}:${J_EVAL_B}:${J_ABLATE} \
             aomt/slurm/run_analysis.sh)

# ── Print summary ──────────────────────────────────────────────────────────
echo "Evaluation & analysis jobs submitted:"
printf "  %-45s Job %s\n" "eval_all_mode_a (all methods, all benchmarks):" "$J_EVAL_A"
printf "  %-45s Job %s\n" "eval_aomt_mixed_mode_b (H sweep + best H):"     "$J_EVAL_B"
printf "  %-45s Job %s\n" "ablate_all (A1-A9):"                            "$J_ABLATE"
printf "  %-45s Job %s\n" "run_analysis (tables + figures):"               "$J_ANALYSIS"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Logs: results/logs/"
echo "Results: results/"
