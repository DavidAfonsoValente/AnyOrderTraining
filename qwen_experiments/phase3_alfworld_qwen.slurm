#!/bin/bash
#SBATCH --job-name=Qwen-Phase3-ALFWorld
#SBATCH --output=logs/qwen_phase3_%A_%a.out
#SBATCH --error=logs/qwen_phase3_%A_%a.err
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=64G
#SBATCH --time=48:00:00 # 48 hours per run (IWM is multi-stage)

#==============================================================================
# Qwen Experiments - Phase 3: ALFWorld Training and Evaluation
#
# This script runs the Qwen-based training and evaluation experiments for Phase 3
# on ALFWorld. It handles both Qwen+IL and Qwen+IWM conditions using a job array.
#
# Qwen Conditions:
# - Qwen+IL (3 seeds)
# - Qwen+IWM (3 seeds)
#
# Total jobs = 2 conditions * 3 seeds = 6
# Submit with: sbatch --array=0-5 qwen_experiments/phase3_alfworld_qwen.slurm
#==============================================================================

# --- Array Job Configuration ---
QWEN_CONDITIONS=(
    "Qwen+IL"
    "Qwen+IWM"
)
SEEDS=(0 1 2)

# Map SLURM_ARRAY_TASK_ID to specific configuration
TASK_ID=$SLURM_ARRAY_TASK_ID
SEED_INDEX=$((TASK_ID % 3))
COND_INDEX=$(((TASK_ID / 3) % 2))

SEED=${SEEDS[$SEED_INDEX]}
CONDITION=${QWEN_CONDITIONS[$COND_INDEX]}

echo "===== JOB CONFIGURATION ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $TASK_ID"
echo "Condition: $CONDITION"
echo "Seed: $SEED"
echo "============================="

# --- User Configuration ---
export PROJECT_ROOT="/home/davidvalente/AnyOrderTraining"

# --- Script Configuration ---
# Assuming the main dFactory venv is activated for all necessary dependencies
VENV_PATH="$PROJECT_ROOT/dFactory/VeOmni/.venv/bin/activate"
if [ ! -f "$VENV_PATH" ]; then
    echo "Error: Virtual environment not found at $VENV_PATH"
    exit 1
fi
source "$VENV_PATH"

# Define common paths
QWEN_MODEL_PATH="$PROJECT_ROOT/models/Qwen-2.5-7B/Qwen/Qwen2-1.5B" # Downloaded by phase1_prepare_data.slurm
ALFWORLD_EXPERT_DATA="$PROJECT_ROOT/qwen_experiments/data/alfworld_expert_train.jsonl"

# Ensure output directories exist
TRAIN_OUTPUT_DIR="$PROJECT_ROOT/results/phase3/${CONDITION}/seed_${SEED}"
mkdir -p "$TRAIN_OUTPUT_DIR"

# Navigate to the qwen_experiments directory to run its scripts
cd "$PROJECT_ROOT/qwen_experiments"

# --- Run Qwen+IL ---
if [ "$CONDITION" == "Qwen+IL" ]; then
    echo "--- Starting Training for Qwen+IL ---"
    python train_qwen_il.py 
      --model_name_or_path "$QWEN_MODEL_PATH" 
      --train_file "$ALFWORLD_EXPERT_DATA" 
      --output_dir "$TRAIN_OUTPUT_DIR" 
      --num_train_epochs 5 
      --per_device_train_batch_size 1 
      --gradient_accumulation_steps 16 
      --learning_rate 2e-5 
      --seed "$SEED" 
      --max_seq_length 1024 
      --bf16 
      --overwrite_output_dir

    if [ $? -ne 0 ]; then
        echo "Qwen+IL training failed for Seed: $SEED"
        exit 1
    fi
    echo "--- Qwen+IL Training complete. ---"

# --- Run Qwen+IWM ---
elif [ "$CONDITION" == "Qwen+IWM" ]; then
    echo "--- Starting Multi-stage Process for Qwen+IWM ---"

    # Stage 1: Fine-tune Qwen with IL to get an agent for rollouts
    IL_MODEL_OUTPUT_DIR="$TRAIN_OUTPUT_DIR/il_stage"
    mkdir -p "$IL_MODEL_OUTPUT_DIR"
    echo "  Stage 1: Initial IL fine-tuning for rollouts..."
    python train_qwen_il.py 
      --model_name_or_path "$QWEN_MODEL_PATH" 
      --train_file "$ALFWORLD_EXPERT_DATA" 
      --output_dir "$IL_MODEL_OUTPUT_DIR" 
      --num_train_epochs 3 
      --per_device_train_batch_size 1 
      --gradient_accumulation_steps 16 
      --learning_rate 2e-5 
      --seed "$SEED" 
      --max_seq_length 1024 
      --bf16 
      --overwrite_output_dir

    if [ $? -ne 0 ]; then
        echo "Qwen+IWM Stage 1 (IL pre-training) failed for Seed: $SEED"
        exit 1
    fi
    echo "  Stage 1 (IL pre-training) complete."

    # Stage 2: Generate rollouts using the IL-tuned model
    ROLLOUT_FILE="$TRAIN_OUTPUT_DIR/rollouts.jsonl"
    mkdir -p "$(dirname "$ROLLOUT_FILE")" # Ensure data directory exists
    echo "  Stage 2: Generating rollouts with IL-tuned model..."
    python rollout_qwen_iwm.py 
      --model_path "$IL_MODEL_OUTPUT_DIR" 
      --output_file "$ROLLOUT_FILE" 
      --num_episodes 100 
      --max_steps_per_episode 50 
      --seed "$SEED"

    if [ $? -ne 0 ]; then
        echo "Qwen+IWM Stage 2 (Rollout generation) failed for Seed: $SEED"
        exit 1
    fi
    echo "  Stage 2 (Rollout generation) complete."

    # Stage 3: Prepare IWM data from expert and rollout data
    IWM_TRAIN_FILE="$TRAIN_OUTPUT_DIR/iwm_train_data.jsonl"
    echo "  Stage 3: Preparing IWM training data..."
    python prepare_iwm_data.py 
      --expert_data_path "$ALFWORLD_EXPERT_DATA" 
      --rollout_data_path "$ROLLOUT_FILE" 
      --output_file "$IWM_TRAIN_FILE"

    if [ $? -ne 0 ]; then
        echo "Qwen+IWM Stage 3 (IWM data prep) failed for Seed: $SEED"
        exit 1
    fi
    echo "  Stage 3 (IWM data prep) complete."

    # Stage 4: Final training with IWM data (using the ORIGINAL Qwen model)
    FINAL_IWM_MODEL_OUTPUT_DIR="$TRAIN_OUTPUT_DIR/final_iwm_model"
    mkdir -p "$FINAL_IWM_MODEL_OUTPUT_DIR"
    echo "  Stage 4: Final IWM training..."
    python train_qwen_iwm.py 
      --model_name_or_path "$QWEN_MODEL_PATH" 
      --train_file "$IWM_TRAIN_FILE" 
      --output_dir "$FINAL_IWM_MODEL_OUTPUT_DIR" 
      --num_train_epochs 5 
      --per_device_train_batch_size 1 
      --gradient_accumulation_steps 16 
      --learning_rate 2e-5 
      --seed "$SEED" 
      --max_seq_length 1024 
      --bf16 
      --overwrite_output_dir

    if [ $? -ne 0 ]; then
        echo "Qwen+IWM Stage 4 (Final IWM training) failed for Seed: $SEED"
        exit 1
    fi
    echo "  Stage 4 (Final IWM training) complete."

    echo "--- Qwen+IWM Multi-stage Process complete. ---"

fi

echo "Phase 3 Qwen Job ($TASK_ID) for condition '$CONDITION' completed successfully."
