#!/bin/bash
#SBATCH --job-name=dFactory-Phase2-MiniGrid
#SBATCH --output=logs/phase2_minigrid_%A_%a.out
#SBATCH --error=logs/phase2_minigrid_%A_%a.err
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH --time=4:00:00 # 4 hours per run

#==============================================================================
# dFactory - Phase 2: MiniGrid Mask Sampling Ablation
#
# This script runs the 72 ablation experiments on MiniGrid using a job array.
# Each job in the array corresponds to one combination of:
# - Masking Condition (8 variations)
# - Environment (3 variations)
# - Seed (3 variations)
#
# Total jobs = 8 * 3 * 3 = 72
# Submit with: sbatch --array=0-71 phase2_minigrid_ablation.slurm
#==============================================================================

# --- Array Job Configuration ---
CONDITIONS=(
    "B-SingleUnit"
    "B-Prefix"
    "AO-p15"
    "AO-p30"
    "AO-p50"
    "AO-p75"
    "AO-ObsOnly"
    "AO-ActOnly"
)
ENVIRONMENTS=(
    "GoToDoor-v0"
    "PickupDist-v0"
    "Unlock-v0"
)
SEEDS=(0 1 2)

# Map SLURM_ARRAY_TASK_ID to specific configuration
TASK_ID=$SLURM_ARRAY_TASK_ID
SEED_INDEX=$((TASK_ID % 3))
ENV_INDEX=$(((TASK_ID / 3) % 3))
COND_INDEX=$(((TASK_ID / 9) % 8))

SEED=${SEEDS[$SEED_INDEX]}
ENVIRONMENT=${ENVIRONMENTS[$ENV_INDEX]}
CONDITION=${CONDITIONS[$COND_INDEX]}

echo "===== JOB CONFIGURATION ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $TASK_ID"
echo "Condition: $CONDITION"
echo "Environment: $ENVIRONMENT"
echo "Seed: $SEED"
echo "============================="

# --- User Configuration ---
export PROJECT_ROOT="/home/davidvalente/AnyOrderTraining"

# --- Script Configuration ---
VENV_PATH="$PROJECT_ROOT/dFactory/VeOmni/.venv/bin/activate"
if [ ! -f "$VENV_PATH" ]; then
    echo "Error: Virtual environment not found at $VENV_PATH"
    exit 1
fi
source "$VENV_PATH"

cd "$PROJECT_ROOT/dFactory"

# Define path to the config file for this specific job
# NOTE: This assumes that config files like 'configs/minigrid/GoToDoor-v0/B-SingleUnit.yaml' exist.
# These will be created in a later step.
CONFIG_FILE="configs/minigrid/${ENVIRONMENT}/${CONDITION}.yaml"

# Define a unique output directory for this specific job
OUTPUT_DIR="$PROJECT_ROOT/results/phase2/${ENVIRONMENT}/${CONDITION}/seed_${SEED}"
mkdir -p "$OUTPUT_DIR"

# --- Run Training ---
echo "--- Starting MiniGrid Training ---"
echo "Config: $CONFIG_FILE"
echo "Output: $OUTPUT_DIR"

# The project spec lists 'tasks/train_any_order.py', which doesn't exist in the repo.
# The original dFactory README uses 'tasks/train_llada2_bd.py'. We assume this is correct.
# We also assume the script or the underlying framework can parse CLI overrides for config values.
PYTHONPATH=$(pwd)/VeOmni:$PYTHONPATH 
    sh train.sh tasks/train_llada2_bd.py "$CONFIG_FILE" 
    --train.output_dir="$OUTPUT_DIR" 
    --train.seed="$SEED"

if [ $? -ne 0 ]; then
  echo "Training failed for Condition: $CONDITION, Environment: $ENVIRONMENT, Seed: $SEED"
  exit 1
fi

echo "--- Training complete. ---"
echo ""

# --- Run Evaluation ---
# The project spec lists evaluation scripts like 'scripts/evaluate_forward_nll.py'.
# This would be run here against the checkpoint in $OUTPUT_DIR.
# This section is a placeholder for now.
echo "--- Starting MiniGrid Evaluation ---"
# CHECKPOINT_PATH=$(find $OUTPUT_DIR/checkpoints -name "hf_ckpt" -type d | head -n 1)
# if [ -n "$CHECKPOINT_PATH" ]; then
#   python scripts/evaluate_forward_nll.py --checkpoint_dir "$CHECKPOINT_PATH"
# else
#   echo "Could not find checkpoint directory in $OUTPUT_DIR"
# fi
echo "Evaluation script not yet implemented. Placeholder."

echo "Phase 2 Job ($TASK_ID) completed successfully."
