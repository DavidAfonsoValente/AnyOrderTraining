#!/bin/bash
# =============================================================================
# submit_pipeline.sh
# Submits all AOMT training jobs targeting the best single hardware config.
# Fixes QOSMaxGRESPerUser issues by avoiding aggressive racing.
# =============================================================================

set -euo pipefail

EMAIL="${1:-}"
if [[ "$1" == "--email" ]]; then
    EMAIL="$2"
fi

SCRIPT_DIR="scripts"
CPUS=8 

echo "=== AOMT Training Pipeline Submission (Stable Mode) ==="
echo "Working dir: $(pwd)"
echo "Email:       ${EMAIL:-none}"
echo ""

# Target: 2 nodes, 2 x H100-96 GPUs each = 4 GPUs total per task.
# Total footprint for 4 parallel tasks = 16 GPUs.
GPU_SPEC="--nodes=2 --gpus-per-node=h100-96:2 --ntasks-per-node=2"

# Helper to clean and join Job IDs
clean_ids() {
    echo "$1" | sed 's/::*/:/g' | sed 's/^://;s/:$//'
}

# Helper to submit a single reliable job
submit_task() {
    local NAME="$1"
    local SCRIPT="$2"
    local DEP_IN="$3"
    local DEPENDENCY=$(clean_ids "$DEP_IN")
    
    local OPTS="--parsable --job-name=$NAME --cpus-per-task=$CPUS $GPU_SPEC"
    if [ -n "$DEPENDENCY" ]; then
        # afterok is preferred for single-target stability
        OPTS="$OPTS --dependency=afterok:$DEPENDENCY"
    fi
    if [ -n "$EMAIL" ]; then
        OPTS="$OPTS --mail-type=BEGIN,END,FAIL --mail-user=$EMAIL"
    fi

    local JID=$(sbatch $OPTS "$SCRIPT")
    echo "$JID"
}

# ---- Step 0: Data preparation (CPU, fast) -----------------------------------
echo "Submitting: data preparation..."
JOB_DATA=$(sbatch --parsable \
    --cpus-per-task=$CPUS \
    ${EMAIL:+--mail-type=END,FAIL} \
    ${EMAIL:+--mail-user=$EMAIL} \
    "$SCRIPT_DIR/01_prepare_data.sh")
echo "  Job ID: $JOB_DATA"

# ---- Step 1: All independent training runs (Parallel) ----------------------
echo ""
echo "Submitting: primary training jobs..."

ID_SFT=$(submit_task "aomt_sft_std" "$SCRIPT_DIR/02_train_sft_standard.sh" "$JOB_DATA")
echo "  Standard SFT ID:     $ID_SFT"

ID_PFX1=$(submit_task "aomt_pfx_s1" "$SCRIPT_DIR/03_train_prefix_s1.sh" "$JOB_DATA")
echo "  Prefix S1 ID:        $ID_PFX1"

ID_ACT=$(submit_task "aomt_act_only" "$SCRIPT_DIR/05_train_aomt_action.sh" "$JOB_DATA")
echo "  AOMT-Action ID:      $ID_ACT"

ID_MIX=$(submit_task "aomt_mixed" "$SCRIPT_DIR/06_train_aomt_mixed.sh" "$JOB_DATA")
echo "  AOMT-Mixed ID:       $ID_MIX"

# ---- Step 2: Prefix SFT Stage 2 (depends on Stage 1) -----------------------
echo ""
echo "Submitting: Prefix SFT Stage 2..."

ID_PFX2=$(submit_task "aomt_pfx_s2" "$SCRIPT_DIR/04_train_prefix_s2.sh" "$ID_PFX1")
echo "  Prefix S2 ID:        $ID_PFX2"

# ---- Step 3: Evaluation (depends on all training) ---------------------------
echo ""
echo "Submitting: evaluation..."

ALL_TRAIN=$(clean_ids "${ID_SFT}:${ID_PFX2}:${ID_ACT}:${ID_MIX}")
JOB_EVAL=$(sbatch --parsable \
    --dependency=afterany:$ALL_TRAIN \
    --cpus-per-task=$CPUS \
    --gpus-per-node=h100-96:1 \
    ${EMAIL:+--mail-type=BEGIN,END,FAIL} \
    ${EMAIL:+--mail-user=$EMAIL} \
    "$SCRIPT_DIR/07_run_eval.sh")
echo "  Evaluation ID:       $JOB_EVAL"

echo ""
echo "=== Submission complete ==="
echo "Monitoring:"
echo "  squeue -u \$USER"
echo ""
