#!/bin/bash
# =============================================================================
# submit_pipeline.sh
# Submits AOMT training jobs in SEQUENTIAL mode to satisfy strict QOS limits.
# Only one task (4 GPUs) will be active in the queue at a time.
# =============================================================================

set -euo pipefail

EMAIL="${1:-}"
if [[ "$1" == "--email" ]]; then
    EMAIL="$2"
fi

SCRIPT_DIR="scripts"
CPUS=8 

echo "=== AOMT Training Pipeline Submission (Sequential Mode) ==="
echo "Working dir: $(pwd)"
echo "Email:       ${EMAIL:-none}"
echo ""

# Target: 1 node, 2 x H100-96 GPUs = 2 GPUs total.
# This satisfies strict QOS limits (usually capped at 2 GPUs for high-end nodes).
GPU_SPEC="--nodes=1 --gpus-per-node=h100-96:2 --ntasks-per-node=2"

clean_ids() {
    echo "$1" | sed 's/::*/:/g' | sed 's/^://;s/:$//'
}

submit_task() {
    local NAME="$1"
    local SCRIPT="$2"
    local DEP_IN="$3"
    local MODE="${4:-afterany}" # Default to afterany so chain continues if one task fails
    local DEPENDENCY=$(clean_ids "$DEP_IN")
    
    local OPTS="--parsable --job-name=$NAME --cpus-per-task=$CPUS $GPU_SPEC"
    if [ -n "$DEPENDENCY" ]; then
        OPTS="$OPTS --dependency=$MODE:$DEPENDENCY"
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

# ---- Step 1: Sequential Training Chain --------------------------------------
# We chain them so only ONE is eligible for resources at a time.
echo ""
echo "Submitting: training jobs (chained sequentially)..."

ID_SFT=$(submit_task "aomt_sft_std" "$SCRIPT_DIR/02_train_sft_standard.sh" "$JOB_DATA" "afterok")
echo "  1. Standard SFT ID:  $ID_SFT"

ID_PFX1=$(submit_task "aomt_pfx_s1" "$SCRIPT_DIR/03_train_prefix_s1.sh" "$ID_SFT")
echo "  2. Prefix S1 ID:     $ID_PFX1 (waits for SFT)"

ID_ACT=$(submit_task "aomt_act_only" "$SCRIPT_DIR/05_train_aomt_action.sh" "$ID_PFX1")
echo "  3. AOMT-Action ID:   $ID_ACT (waits for PFX1)"

ID_MIX=$(submit_task "aomt_mixed" "$SCRIPT_DIR/06_train_aomt_mixed.sh" "$ID_ACT")
echo "  4. AOMT-Mixed ID:    $ID_MIX (waits for ACT)"

# ---- Step 2: Prefix SFT Stage 2 (depends specifically on S1 success) --------
echo ""
echo "Submitting: Prefix SFT Stage 2..."
# This must use afterok because it needs the specific checkpoint from S1
ID_PFX2=$(submit_task "aomt_pfx_s2" "$SCRIPT_DIR/04_train_prefix_s2.sh" "$ID_PFX1" "afterok")
echo "  Prefix S2 ID:        $ID_PFX2"

# ---- Step 3: Evaluation (depends on the end of the chain) -------------------
echo ""
echo "Submitting: evaluation..."

# Eval waits for the last training job in the chain AND the Stage 2 job
ALL_TRAIN=$(clean_ids "${ID_MIX}:${ID_PFX2}")
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
echo "Note: Jobs are chained. Only one training task will request GPUs at a time"
echo "to ensure you stay within your cluster's QOS limits."
echo ""
