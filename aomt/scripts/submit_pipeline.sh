#!/bin/bash
# =============================================================================
# submit_pipeline.sh
# Submits all AOMT training jobs with a refined "Best Available" Race strategy.
# Focuses on high-probability nodes and fixes CPU/Dependency issues.
# =============================================================================

set -euo pipefail

EMAIL="${1:-}"
if [[ "$1" == "--email" ]]; then
    EMAIL="$2"
fi

SCRIPT_DIR="scripts"
CPUS=8 # Explicitly set to match script headers

echo "=== AOMT Training Pipeline Submission (Refined Race) ==="
echo "Working dir: $(pwd)"
echo "Email:       ${EMAIL:-none}"
echo ""

clean_ids() {
    echo "$1" | sed 's/::*/:/g' | sed 's/^://;s/:$//'
}

# Helper to submit a race of jobs for the best available hardware
submit_race() {
    local NAME="$1"
    local SCRIPT="$2"
    local DEP_IN="$3"
    local DEPENDENCY=$(clean_ids "$DEP_IN")
    
    local OPTS="--parsable --job-name=$NAME --cpus-per-task=$CPUS"
    if [ -n "$DEPENDENCY" ]; then
        OPTS="$OPTS --dependency=afterany:$DEPENDENCY"
    fi
    if [ -n "$EMAIL" ]; then
        OPTS="$OPTS --mail-type=BEGIN,END,FAIL --mail-user=$EMAIL"
    fi

    # Variant 1: Native H100-96 (2 nodes, 4 GPUs total) - Highest priority
    local J1=$(sbatch $OPTS --nodes=2 --gpus-per-node=h100-96:2 --ntasks-per-node=2 "$SCRIPT" 2>/dev/null || echo "")
    
    # Variant 2: H100-47 MIG (2 nodes, 8 GPUs total) - High availability
    local J2=$(sbatch $OPTS --nodes=2 --gpus-per-node=h100-47:4 --ntasks-per-node=4 "$SCRIPT" 2>/dev/null || echo "")

    # Variant 3: H200-141 (1 node, 4 GPUs) - Best if available
    local J3=$(sbatch $OPTS --nodes=1 --gpus-per-node=h200-141:4 --ntasks-per-node=4 "$SCRIPT" 2>/dev/null || echo "")

    echo "${J1}:${J2}:${J3}"
}

# ---- Step 0: Data preparation (CPU, fast) -----------------------------------
echo "Submitting: data preparation..."
JOB_DATA=$(sbatch --parsable \
    --cpus-per-task=$CPUS \
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
echo "Submitting: evaluation (depends on all training races)..."

ALL_TRAIN=$(clean_ids "${IDS_SFT}:${IDS_PFX2}:${IDS_ACT}:${IDS_MIX}")
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
echo "Note: Downstream jobs (Stage 2 and Eval) will check for checkpoints"
echo "to ensure the previous stage actually succeeded."
echo ""
