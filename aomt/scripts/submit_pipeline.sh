#!/bin/bash
# =============================================================================
# submit_pipeline.sh
# Submits all AOMT training jobs with correct Slurm dependencies.
# Run from the aomt/ directory.
# Usage: bash scripts/submit_pipeline.sh [--email user@comp.nus.edu.sg]
# =============================================================================

set -euo pipefail

EMAIL="${1:-}"
if [[ "$1" == "--email" ]]; then
    EMAIL="$2"
fi

SCRIPT_DIR="scripts"

echo "=== AOMT Training Pipeline Submission ==="
echo "Working dir: $(pwd)"
echo "Email:       ${EMAIL:-none}"
echo ""

# ---- Step 0: Data preparation (CPU, fast) -----------------------------------
echo "Submitting: data preparation..."
# Use 01_prepare_data.sh which is the Slurm-wrapped version
JOB_DATA=$(sbatch --parsable \
    ${EMAIL:+--mail-type=END,FAIL} \
    ${EMAIL:+--mail-user=$EMAIL} \
    "$SCRIPT_DIR/01_prepare_data.sh")

echo "  Job ID: $JOB_DATA"

# ---- Step 1: All independent training runs (can run in parallel) ------------
echo ""
echo "Submitting: training jobs (parallel, depend on data)..."

JOB_SFT=$(sbatch --parsable \
    --dependency=afterok:$JOB_DATA \
    ${EMAIL:+--mail-type=BEGIN,END,FAIL} \
    ${EMAIL:+--mail-user=$EMAIL} \
    "$SCRIPT_DIR/02_train_sft_standard.sh")
echo "  Standard SFT:        $JOB_SFT"

JOB_PFX1=$(sbatch --parsable \
    --dependency=afterok:$JOB_DATA \
    ${EMAIL:+--mail-type=BEGIN,END,FAIL} \
    ${EMAIL:+--mail-user=$EMAIL} \
    "$SCRIPT_DIR/03_train_prefix_s1.sh")
echo "  Prefix SFT Stage 1:  $JOB_PFX1"

JOB_ACT=$(sbatch --parsable \
    --dependency=afterok:$JOB_DATA \
    ${EMAIL:+--mail-type=BEGIN,END,FAIL} \
    ${EMAIL:+--mail-user=$EMAIL} \
    "$SCRIPT_DIR/05_train_aomt_action.sh")
echo "  AOMT-Action-Only:    $JOB_ACT"

JOB_MIX=$(sbatch --parsable \
    --dependency=afterok:$JOB_DATA \
    ${EMAIL:+--mail-type=BEGIN,END,FAIL} \
    ${EMAIL:+--mail-user=$EMAIL} \
    "$SCRIPT_DIR/06_train_aomt_mixed.sh")
echo "  AOMT-Mixed:          $JOB_MIX"

# ---- Step 2: Prefix SFT Stage 2 (depends on Stage 1) -----------------------
echo ""
echo "Submitting: Prefix SFT Stage 2 (depends on Stage 1)..."

JOB_PFX2=$(sbatch --parsable \
    --dependency=afterok:$JOB_PFX1 \
    ${EMAIL:+--mail-type=BEGIN,END,FAIL} \
    ${EMAIL:+--mail-user=$EMAIL} \
    "$SCRIPT_DIR/04_train_prefix_s2.sh")
echo "  Prefix SFT Stage 2:  $JOB_PFX2"

# ---- Step 3: Evaluation (depends on all training) ---------------------------
echo ""
echo "Submitting: evaluation (depends on all training)..."

ALL_TRAIN="${JOB_SFT}:${JOB_PFX2}:${JOB_ACT}:${JOB_MIX}"
JOB_EVAL=$(sbatch --parsable \
    --dependency=afterok:$ALL_TRAIN \
    ${EMAIL:+--mail-type=BEGIN,END,FAIL} \
    ${EMAIL:+--mail-user=$EMAIL} \
    "$SCRIPT_DIR/07_run_eval.sh")
echo "  Evaluation:          $JOB_EVAL"

echo ""
echo "=== Submission complete ==="
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  sacct -j $JOB_SFT,$JOB_PFX1,$JOB_PFX2,$JOB_ACT,$JOB_MIX,$JOB_EVAL"
echo ""
echo "Cancel everything with:"
echo "  scancel $JOB_DATA $JOB_SFT $JOB_PFX1 $JOB_PFX2 $JOB_ACT $JOB_MIX $JOB_EVAL"
