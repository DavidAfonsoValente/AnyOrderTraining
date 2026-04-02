#!/bin/bash
# =============================================================================
# submit_pipeline.sh
# Phase 1: Hyperparameter Search (Ablation on mask_prob)
# Intermediate: Auto-Selection of winning probability
# Phase 2: Main Benchmarking (Optimal Sequential Chain)
# =============================================================================

set -euo pipefail

EMAIL="${1:-}"
if [[ "$1" == "--email" ]]; then
    EMAIL="$2"
fi

SCRIPT_DIR="scripts"
CPUS=8 
GPU_SPEC="--nodes=1 --gpus-per-node=h100-96:2 --ntasks-per-node=2"

echo "=== AOMT Training Pipeline Submission (Fully Automated) ==="
echo "Working dir: $(pwd)"
echo ""

# Helper to clean and join Job IDs
clean_ids() {
    echo "$1" | sed 's/::*/:/g' | sed 's/^://;s/:$//'
}

# ---- Step 0: Data preparation -----------------------------------------------
echo "Submitting: data preparation..."
JOB_DATA=$(sbatch --parsable --cpus-per-task=$CPUS "$SCRIPT_DIR/01_prepare_data.sh")
JOB_ABL_DATA=$(sbatch --parsable --dependency=afterok:$JOB_DATA --cpus-per-task=$CPUS "$SCRIPT_DIR/01b_prepare_ablation_data.sh")

# ---- Step 1: Phase 1 - Ablation Sweep (Sequential) --------------------------
echo "Submitting: Phase 1 Ablation Sweep (0.15, 0.25, 0.40, 0.50)..."

PREV_ABL=$JOB_ABL_DATA
ABL_IDS=""
for P in 0.15 0.25 0.40 0.50; do
    JID=$(sbatch --parsable --dependency=afterany:$PREV_ABL "$SCRIPT_DIR/run_single_ablation.sh" "$P")
    ABL_IDS="${ABL_IDS}:${JID}"
    PREV_ABL=$JID
done
echo "  Ablation IDs: $ABL_IDS"

# ---- Step 2: Auto-Selection -------------------------------------------------
# This job runs after the LAST ablation is finished.
echo "Submitting: automatic hyperparameter selection..."
CLEAN_ABL=$(clean_ids "$ABL_IDS")
JOB_UPDATE=$(sbatch --parsable --dependency=afterany:$CLEAN_ABL "$SCRIPT_DIR/apply_ablation_winner.sh")
echo "  Selection Job ID: $JOB_UPDATE"

# ---- Step 3: Phase 2 - Main Experiments -------------------------------------
# These depend on the updater job finishing.
echo ""
echo "Submitting: Phase 2 Main Benchmarking (will use optimal settings)..."

submit_task() {
    local NAME="$1"; local SCRIPT="$2"; local DEP="$3"; local MODE="${4:-afterany}"
    sbatch --parsable --job-name=$NAME --cpus-per-task=$CPUS $GPU_SPEC --dependency=$MODE:$DEP "$SCRIPT"
}

# Chain the experiments sequentially to stay within QOS
ID_SFT=$(submit_task "aomt_sft_std" "$SCRIPT_DIR/02_train_sft_standard.sh" "$JOB_UPDATE")
ID_PFX1=$(submit_task "aomt_pfx_s1" "$SCRIPT_DIR/03_train_prefix_s1.sh" "$ID_SFT")
ID_ACT=$(submit_task "aomt_act_only" "$SCRIPT_DIR/05_train_aomt_action.sh" "$ID_PFX1")
ID_MIX=$(submit_task "aomt_mixed" "$SCRIPT_DIR/06_train_aomt_mixed.sh" "$ID_ACT")
ID_PFX2=$(submit_task "aomt_pfx_s2" "$SCRIPT_DIR/04_train_prefix_s2.sh" "$ID_PFX1" "afterok")

# ---- Step 4: Final Evaluation -----------------------------------------------
ALL_TRAIN="${ID_MIX}:${ID_PFX2}"
JOB_EVAL=$(sbatch --parsable --dependency=afterany:$ALL_TRAIN --cpus-per-task=$CPUS --gpus-per-node=h100-96:1 "$SCRIPT_DIR/07_run_eval.sh")

echo ""
echo "=== Submission complete ==="
echo "Pipeline Strategy:"
echo "  1. Run 4-way hyperparameter sweep on ALFWorld subset."
echo "  2. Automatically select best mask_prob and update YAML configs."
echo "  3. Run full benchmarking suite with optimal hyperparameters."
echo "  4. Generate final evaluation table."
echo ""
echo "Monitor: squeue -u \$USER"
