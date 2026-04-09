#!/bin/bash
# slurm/run_part2_aomt_eval_and_analysis.sh
# Submits final AOMT evaluation, ablations, NLL, and analysis.
# Reads Part 1 job IDs from slurm/.part1_job_ids.
set -euo pipefail

PROJECT_DIR="/home/d/dvalente/AnyOrderTraining"
IDS_FILE="$PROJECT_DIR/slurm/.part1_job_ids"

echo "═══════════════════════════════════════════════════════════"
echo "AOMT Part 2: Final Eval, Ablations, NLL, Analysis"
echo "═══════════════════════════════════════════════════════════"

if [ ! -f "$IDS_FILE" ]; then
  echo "ERROR: $IDS_FILE not found."
  echo "Run bash slurm/run_part1_baselines_and_sweeps.sh first."
  exit 1
fi

ALL_PART1=$(grep -v '^$' "$IDS_FILE" | tr '\n' ':' | sed 's/:$//')
DEP_PART1="--dependency=afterok:${ALL_PART1}"

# ── 1. Main Evaluation ─────────────────────────────────────────
J_EVAL=$(sbatch --parsable $DEP_PART1 slurm/eval_all.sh)

# ── 2. K-Denoising Sweep (Inference ablation) ─────────────────
J_KSWEEP=$(sbatch --parsable $DEP_PART1 slurm/eval_ksweep.sh)

# ── 3. NLL computation (aomt_mixed only) ───────────────────────
J_NLL=$(sbatch --parsable --dependency=afterok:${ALL_PART1} slurm/compute_nll.sh)

# ── 4. Ablation processing (A1-A6) ─────────────────────────────
J_ABLATE=$(sbatch --parsable \
           --dependency=afterok:${J_EVAL}:${J_KSWEEP} \
           slurm/ablate_all.sh)

# ── 5. Final Analysis ──────────────────────────────────────────
J_ANALYSIS=$(sbatch --parsable \
             --dependency=afterok:${J_EVAL}:${J_KSWEEP}:${J_ABLATE}:${J_NLL} \
             slurm/run_analysis.sh)

echo "Part 2 jobs submitted:"
printf "  %-45s Job %s\n" "eval_all (main methods):"      "$J_EVAL"
printf "  %-45s Job %s\n" "eval_ksweep (K ablation):"     "$J_KSWEEP"
printf "  %-45s Job %s\n" "ablate_all (A1-A6):"           "$J_ABLATE"
printf "  %-45s Job %s\n" "compute_nll (aomt_mixed):"     "$J_NLL"
printf "  %-45s Job %s\n" "run_analysis (final results):" "$J_ANALYSIS"
echo ""
echo "All results will be in: results/"
