#!/bin/bash
# slurm/run_part2_aomt_eval_and_analysis.sh
# Submits final AOMT evaluation, ablations, NLL, and analysis.
# Reads Part 1 job IDs from slurm/.part1_job_ids.
# Usage: bash slurm/run_part2_aomt_eval_and_analysis.sh
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
IDS_FILE="$PROJECT_DIR/slurm/.part1_job_ids"

echo "═══════════════════════════════════════════════════════════"
echo "AOMT Part 2: Final Eval, Ablations, NLL, Analysis"
echo "═══════════════════════════════════════════════════════════"

if [ ! -f "$IDS_FILE" ]; then
  echo "ERROR: $IDS_FILE not found."
  echo "Run bash slurm/run_part1_baselines_and_sweeps.sh first."
  exit 1
fi

# Read all Part 1 job IDs
ALL_PART1=$(grep -v '^$' "$IDS_FILE" | tr '\n' ':' | sed 's/:$//')
echo "Depending on Part 1 jobs: $ALL_PART1"
echo ""

DEP_PART1="--dependency=afterok:${ALL_PART1}"

# ── Evaluation: all methods, all 3 benchmarks ───────────────────
# All methods now use identical unified inference logic.
J_EVAL=$(sbatch --parsable $DEP_PART1 aomt/slurm/eval_all.sh)

# ── Ablation evaluations (A1-A6) ────────────────────────────────
J_ABLATE=$(sbatch --parsable \
           --dependency=afterok:${ALL_PART1}:${J_EVAL} \
           aomt/slurm/ablate_all.sh)

# ── NLL computation (aomt_mixed only) ───────────────────────────
J_NLL=$(sbatch --parsable \
        --dependency=afterok:${ALL_PART1} \
        aomt/slurm/compute_nll.sh)

# ── Analysis: tables + figures ──────────────────────────────────
J_ANALYSIS=$(sbatch --parsable \
             --dependency=afterok:${J_EVAL}:${J_ABLATE}:${J_NLL} \
             aomt/slurm/run_analysis.sh)

echo "Part 2 jobs submitted:"
printf "  %-45s Job %s\n" "eval_all (all methods, 3 benchmarks):" "$J_EVAL"
printf "  %-45s Job %s\n" "ablate_all (A1-A6):"                  "$J_ABLATE"
printf "  %-45s Job %s\n" "compute_nll (aomt_mixed):"            "$J_NLL"
printf "  %-45s Job %s\n" "run_analysis (tables + figures):"     "$J_ANALYSIS"
echo ""
echo "When complete, all results will be in: results/"
echo "Monitor: squeue -u \$USER | grep aomt"
