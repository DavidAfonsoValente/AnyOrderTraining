#!/bin/bash
# =============================================================================
# submit_full_pipeline.slurm
#
# MASTER CONTROLLER SCRIPT for AOMT Project
# 
# This script manages the entire lifecycle:
# 1. Model Preparation (Merging MoE)
# 2. Data Preparation (Length analysis & JSONL generation)
# 3. Phase 1: Hyperparameter Ablation (Sequential Sweep)
# 4. Auto-Selection: Picking the best mask_prob
# 5. Phase 2: Main Benchmarking (Optimal Sequential Chain)
# 6. Final Evaluation: Success rates, NLL-obs, and Robustness
#
# Usage: sbatch submit_full_pipeline.slurm
# =============================================================================
#SBATCH --job-name=aomt_master
#SBATCH --output=/home/davidvalente/AnyOrderTraining/aomt/logs/master_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00       # Launcher script only needs a few minutes
#SBATCH --partition=normal     # Runs on a CPU node

# Support for optional email notifications
# SBATCH --mail-type=ALL
# SBATCH --mail-user=your_email@comp.nus.edu.sg

set -euo pipefail
mkdir -p logs checkpoints weights results data/cache

# Use absolute path for logs directory to avoid ambiguity
LOGS_DIR="/home/davidvalente/AnyOrderTraining/aomt/logs"
mkdir -p "$LOGS_DIR"

SCRIPTS_PATH="scripts"

echo "=== AOMT Master Pipeline Started [$(date)] ==="
echo "Working directory: $(pwd)"
echo "Logs will be at: $LOGS_DIR"

# ---- PRE-FLIGHT CHECKS ------------------------------------------------------
if [ ! -f "activate_env.sh" ]; then
    echo "ERROR: activate_env.sh not found. Ensure you are running this from the 'aomt/' directory."
    exit 1
fi

# We source the environment to ensure paths are correct for the submission logic
source activate_env.sh

if [ ! -d "dFactory" ]; then
    echo "ERROR: dFactory submodule missing. Run 'bash full_setup.sh' first."
    exit 1
fi

# Helper to clean and join Job IDs for dependencies
clean_ids() {
    echo "$1" | sed 's/::*/:/g' | sed 's/^://;s/:$//'
}

# ---- STEP -1: Model Preparation (MoE Merging) -------------------------------
echo "[$(date)] Submitting Step -1: Model Preparation..."
JOB_PREP_MODEL=$(sbatch --parsable --output="$LOGS_DIR/prepare_model_%j.log" "$SCRIPTS_PATH/prepare_model.sh")
echo "  Job ID: $JOB_PREP_MODEL"

# ---- STEP 0: Data Preparation -----------------------------------------------
echo "[$(date)] Submitting Step 0: Data Preparation..."
# Depends on Model Preparation (for tokenizer files)
JOB_DATA=$(sbatch --parsable --output="$LOGS_DIR/01_prepare_data_%j.log" --dependency=afterok:$JOB_PREP_MODEL "$SCRIPTS_PATH/01_prepare_data.sh")
JOB_ABL_DATA=$(sbatch --parsable --output="$LOGS_DIR/01b_prepare_ablation_%j.log" --dependency=afterok:$JOB_DATA "$SCRIPTS_PATH/01b_prepare_ablation_data.sh")
echo "  Data Job ID: $JOB_DATA"
echo "  Ablation Data Job ID: $JOB_ABL_DATA"

# ---- STEP 1: Phase 1 - Ablation Sweep (Sequential) --------------------------
echo "[$(date)] Submitting Phase 1: Hyperparameter Sweep (mask_prob)..."
# Chained sequentially to respect cluster QOS
PREV_ABL=$JOB_ABL_DATA
ABL_IDS=""
for P in 0.15 0.25 0.40 0.50; do
    JID=$(sbatch --parsable --output="$LOGS_DIR/ablation_${P}_%j.log" --dependency=afterany:$PREV_ABL "$SCRIPTS_PATH/run_single_ablation.sh" "$P")
    ABL_IDS="${ABL_IDS}:${JID}"
    PREV_ABL=$JID
    echo "  Queued ablation p=$P (Job: $JID)"
done

# ---- STEP 2: Automatic Hyperparameter Selection -----------------------------
echo "[$(date)] Submitting Step 2: Auto-Selection..."
CLEAN_ABL=$(clean_ids "$ABL_IDS")
JOB_UPDATE=$(sbatch --parsable --output="$LOGS_DIR/ablation_update_%j.log" --dependency=afterany:$CLEAN_ABL "$SCRIPTS_PATH/apply_ablation_winner.sh")
echo "  Update Job ID: $JOB_UPDATE"

# ---- STEP 3: Phase 2 - Main Benchmarking ------------------------------------
echo "[$(date)] Submitting Phase 2: Main Benchmarking (Sequential Chain)..."

# Helper to submit tasks using their internal SBATCH headers
submit_task() {
    local SCRIPT="$1"; local DEP="$2"; local LOG_NAME="$3"; local MODE="${4:-afterany}"
    sbatch --parsable --output="$LOGS_DIR/${LOG_NAME}_%j.log" --dependency=$MODE:$DEP "$SCRIPT"
}

# Chain the experiments sequentially to stay within QOS
ID_SFT=$(submit_task "$SCRIPTS_PATH/02_train_sft_standard.sh" "$JOB_UPDATE" "02_sft_standard")
echo "  Queued Standard SFT (Job: $ID_SFT)"

ID_PFX1=$(submit_task "$SCRIPTS_PATH/03_train_prefix_s1.sh" "$ID_SFT" "03_prefix_s1")
echo "  Queued Prefix S1 (Job: $ID_PFX1)"

ID_ACT=$(submit_task "$SCRIPTS_PATH/05_train_aomt_action.sh" "$ID_PFX1" "05_aomt_action")
echo "  Queued AOMT-Action (Job: $ID_ACT)"

ID_MIX=$(submit_task "$SCRIPTS_PATH/06_train_aomt_mixed.sh" "$ID_ACT" "06_aomt_mixed")
echo "  Queued AOMT-Mixed (Job: $ID_MIX)"

# Prefix S2 depends on S1 but can run in parallel with others if resources allow,
# but we chain it to ID_PFX1 to ensure S1 finished ok.
ID_PFX2=$(submit_task "$SCRIPTS_PATH/04_train_prefix_s2.sh" "$ID_PFX1" "04_prefix_s2" "afterok")
echo "  Queued Prefix S2 (Job: $ID_PFX2)"

# ---- STEP 4: Final Evaluation -----------------------------------------------
echo "[$(date)] Submitting Step 4: Final Evaluation..."
ALL_TRAIN="${ID_MIX}:${ID_PFX2}"
JOB_EVAL=$(sbatch --parsable --output="$LOGS_DIR/07_eval_%j.log" --dependency=afterany:$ALL_TRAIN "$SCRIPTS_PATH/07_run_eval.sh")

echo "  Final Eval Job ID: $JOB_EVAL"

echo ""
echo "=== All components queued successfully ==="
echo "Pipeline Strategy:"
echo "  1. Prepare Model & Data (Sequential)"
echo "  2. Run 4-way ablation sweep (Sequential)"
echo "  3. Select best mask_prob automatically"
echo "  4. Run 5 main experiments (Sequential Chain)"
echo "  5. Generate final results table"
echo ""
echo "Monitor progress: squeue -u $USER"
echo "Master log: logs/master_${SLURM_JOB_ID}.log"
echo "Final results will be in results/summary_table.txt"
