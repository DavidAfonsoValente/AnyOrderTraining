#!/bin/bash
# slurm/run_part2_aomt_eval_and_analysis.sh
# Submits final AOMT evaluation, ablations, NLL, and analysis.
# Reads Part 1 job IDs from slurm/.part1_job_ids.
# Usage: bash slurm/run_part2_aomt_eval_and_analysis.sh
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
# CLUSTER_TODO: set correct scratch path
SCRATCH=$HOME/scratch
IDS_FILE="$PROJECT_DIR/slurm/.part1_job_ids"

echo "═══════════════════════════════════════════════════════════"
echo "AOMT Part 2: Final Eval, Ablations, NLL, Analysis"
echo "═══════════════════════════════════════════════════════════"

if [ ! -f "$IDS_FILE" ]; then
  echo "ERROR: $IDS_FILE not found."
  echo "Run bash slurm/run_part1_baselines_and_sweeps.sh first."
  exit 1
fi

# Read all Part 1 job IDs into one colon-separated dependency string
ALL_PART1=$(grep -v '^$' "$IDS_FILE" | tr '\n' ':' | sed 's/:$//')
echo "Depending on Part 1 jobs: $ALL_PART1"
echo ""

DEP_PART1="--dependency=afterok:${ALL_PART1}"

# ── Evaluation: all baselines, Mode A, all 3 benchmarks ───────────────────
J_EVAL_A=$(sbatch --parsable $DEP_PART1 aomt/slurm/eval_all_mode_a.sh)

# ── Evaluation: aomt_mixed Mode B, H sweep {1,2,3,5}, all 3 benchmarks ────
J_EVAL_B=$(sbatch --parsable $DEP_PART1 aomt/slurm/eval_aomt_mixed_mode_b.sh)

# ── All ablation evaluations (depends on both Mode A and Mode B results) ─
J_ABLATE=$(sbatch --parsable \
           --dependency=afterok:${J_EVAL_A}:${J_EVAL_B} \
           aomt/slurm/ablate_all.sh)

# ── NLL computation (aomt_mixed Mode A and Mode B, per-checkpoint) ─────────
J_NLL=$(sbatch --parsable \
        --dependency=afterok:${J_EVAL_A}:${J_EVAL_B} \
        aomt/slurm/compute_nll.sh)

# ── Analysis: all tables + all figures (depend on everything above) ─────────
J_ANALYSIS=$(sbatch --parsable \
             --dependency=afterok:${J_EVAL_A}:${J_EVAL_B}:${J_ABLATE}:${J_NLL} \
             aomt/slurm/run_analysis.sh)

echo "Part 2 jobs submitted:"
printf "  %-45s Job %s\n" "eval_all_mode_a (all methods, 3 benchmarks):" "$J_EVAL_A"
printf "  %-45s Job %s\n" "eval_aomt_mixed_mode_b (H sweep):"            "$J_EVAL_B"
printf "  %-45s Job %s\n" "ablate_all (A1-A6):"                          "$J_ABLATE"
printf "  %-45s Job %s\n" "compute_nll (Mode A + B, per-checkpoint):"    "$J_NLL"
printf "  %-45s Job %s\n" "run_analysis (4 tables + 7 figures):"         "$J_ANALYSIS"
echo ""
echo "When complete, all results will be in: results/"
echo "Monitor: squeue -u \$USER | grep aomt"
echo "Logs: $SCRATCH/aomt_logs/"
