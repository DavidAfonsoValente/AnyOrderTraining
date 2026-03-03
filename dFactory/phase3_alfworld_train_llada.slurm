#!/bin/bash
#SBATCH --job-name=dFactory-Phase3-Train-LLaDA
#SBATCH --output=logs/phase3_train_llada_%A_%a.out
#SBATCH --error=logs/phase3_train_llada_%A_%a.err
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=64G
#SBATCH --time=24:00:00 # 24 hours per run

#==============================================================================
# dFactory - Phase 3: ALFWorld Training (LLaDA Conditions)
#
# This script runs the LLaDA-based training experiments for Phase 3 on ALFWorld.
# It uses a job array to manage the 9 LLaDA training runs.
#
# LLaDA Conditions:
# - LLaDA+B-SU
# - LLaDA+B-Pfx
# - LLaDA+AOMT-Best
#
# Total jobs = 3 conditions * 3 seeds = 9
# Submit with: sbatch --array=0-8 phase3_alfworld_train_llada.slurm
#
# NOTE: The Qwen+IL and Qwen+IWM conditions are NOT handled by this script as
# they require a separate training pipeline not included in the dFactory repo.
#==============================================================================

# --- Array Job Configuration ---
LLADA_CONDITIONS=(
    "LLaDA+B-SU"
    "LLaDA+B-Pfx"
    "LLaDA+AOMT-Best"
)
SEEDS=(0 1 2)

# Map SLURM_ARRAY_TASK_ID to specific configuration
TASK_ID=$SLURM_ARRAY_TASK_ID
SEED_INDEX=$((TASK_ID % 3))
COND_INDEX=$(((TASK_ID / 3) % 3))

SEED=${SEEDS[$SEED_INDEX]}
CONDITION=${LLADA_CONDITIONS[$COND_INDEX]}

echo "===== JOB CONFIGURATION ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $TASK_ID"
echo "Condition: $CONDITION"
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
# NOTE: Assumes configs like 'configs/alfworld/LLaDA+B-SU.yaml' will be created.
CONFIG_FILE="configs/alfworld/${CONDITION}.yaml"

# Define a unique output directory for this specific job
OUTPUT_DIR="$PROJECT_ROOT/results/phase3/${CONDITION}/seed_${SEED}"
mkdir -p "$OUTPUT_DIR"

# --- Run Training ---
echo "--- Starting ALFWorld Training for $CONDITION ---"
echo "Config: $CONFIG_FILE"
echo "Output: $OUTPUT_DIR"

# Adapting the training command for Phase 3
PYTHONPATH=$(pwd)/VeOmni:$PYTHONPATH 
    sh train.sh tasks/train_llada2_bd.py "$CONFIG_FILE" 
    --train.output_dir="$OUTPUT_DIR" 
    --train.seed="$SEED"

if [ $? -ne 0 ]; then
  echo "Training failed for Condition: $CONDITION, Seed: $SEED"
  exit 1
fi

echo "--- Training complete. ---"
echo ""

echo "Phase 3 Training Job ($TASK_ID) for LLaDA condition '$CONDITION' completed successfully."
