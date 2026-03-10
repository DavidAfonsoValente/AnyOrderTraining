#!/bin/bash
# =============================================================================
# submit_pipeline.sh
# Submits all AOMT training jobs with "Best Available" GPU Race strategy.
# Run from the aomt/ directory.
# =============================================================================

set -euo pipefail

EMAIL="${1:-}"
if [[ "$1" == "--email" ]]; then
    EMAIL="$2"
fi

SCRIPT_DIR="scripts"

echo "=== AOMT Training Pipeline Submission (Best-Available Race) ==="
echo "Working dir: $(pwd)"
echo "Email:       ${EMAIL:-none}"
echo ""

# Helper to submit a race of jobs for different GPU types
# Usage: submit_race <job_name> <script_path> <dependency_str>
submit_race() {
    local NAME="$1"
    local SCRIPT="$2"
    local DEPENDENCY="$3"
    
    local OPTS="--parsable --job-name=$NAME"
    if [ -n "$DEPENDENCY" ]; then
        # Use afterany: if we use afterok, cancelled jobs in the race will break it
        OPTS="$OPTS --dependency=afterany:$DEPENDENCY"
    fi
    if [ -n "$EMAIL" ]; then
        OPTS="$OPTS --mail-type=BEGIN,END,FAIL --mail-user=$EMAIL"
    fi

    # Variant 1: H100-96 (2 GPUs, most common high-end)
    local J1=$(sbatch $OPTS --gpus-per-node=h100-96:2 --ntasks-per-node=2 "$SCRIPT")
    # Variant 2: H100-47 (4 GPUs, MIG partition)
    local J2=$(sbatch $OPTS --gpus-per-node=h100-47:4 --ntasks-per-node=4 "$SCRIPT")
    # Variant 3: A100-80 (4 GPUs, high VRAM fallback)
    local J3=$(sbatch $OPTS --gpus-per-node=a100-80:4 --ntasks-per-node=4 "$SCRIPT")
    # Variant 4: H200-141 (4 GPUs, best option if available)
    local J4=$(sbatch $OPTS --gpus-per-node=h200-141:4 --ntasks-per-node=4 "$SCRIPT")

    # Return colon-separated list for subsequent dependencies
    echo "$J1:$J2:$J3:$J4"
}

# ---- Step 0: Data preparation (CPU, fast) -----------------------------------
echo "Submitting: data preparation..."
JOB_DATA=$(sbatch --parsable \
    ${EMAIL:+--mail-type=END,FAIL} \
    ${EMAIL:+--mail-user=$EMAIL} \
    "$SCRIPT_DIR/01_prepare_data.sh")
echo "  Job ID: $JOB_DATA"

# ---- Step 1: All independent training runs (Race Mode) ----------------------
echo ""
echo "Submitting: training jobs (parallel races, depend on data)..."

IDS_SFT=$(submit_race "aomt_sft_std" "$SCRIPT_DIR/02_train_sft_standard.sh" "$JOB_DATA")
echo "  Standard SFT IDs:    $IDS_SFT"

IDS_PFX1=$(submit_race "aomt_pfx_s1" "$SCRIPT_DIR/03_train_prefix_s1.sh" "$JOB_DATA")
echo "  Prefix S1 IDs:       $IDS_PFX1"

IDS_ACT=$(submit_race "aomt_act_only" "$SCRIPT_DIR/05_train_aomt_action.sh" "$JOB_DATA")
echo "  AOMT-Action IDs:     $IDS_ACT"

IDS_MIX=$(submit_race "aomt_mixed" "$SCRIPT_DIR/06_train_aomt_mixed.sh" "$JOB_DATA")
echo "  AOMT-Mixed IDs:      $IDS_MIX"

# ---- Step 2: Prefix SFT Stage 2 (depends on any Stage 1 completion) --------
echo ""
echo "Submitting: Prefix SFT Stage 2 (race, depends on Stage 1)..."

IDS_PFX2=$(submit_race "aomt_pfx_s2" "$SCRIPT_DIR/04_train_prefix_s2.sh" "$IDS_PFX1")
echo "  Prefix S2 IDs:       $IDS_PFX2"

# ---- Step 3: Evaluation (depends on all training) ---------------------------
echo ""
echo "Submitting: evaluation (depends on all successful training)..."

# Eval only needs 1 GPU, no race needed usually, but we pick the best type
ALL_TRAIN="${IDS_SFT}:${IDS_PFX2}:${IDS_ACT}:${IDS_MIX}"
JOB_EVAL=$(sbatch --parsable \
    --dependency=afterany:$ALL_TRAIN \
    --gpus-per-node=h100-96:1 \
    ${EMAIL:+--mail-type=BEGIN,END,FAIL} \
    ${EMAIL:+--mail-user=$EMAIL} \
    "$SCRIPT_DIR/07_run_eval.sh")
echo "  Evaluation:          $JOB_EVAL"

echo ""
echo "=== Submission complete ==="
echo "Note: Jobs are racing. The first variant of each task to start will"
echo "automatically cancel its siblings to maximize cluster efficiency."
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo ""
