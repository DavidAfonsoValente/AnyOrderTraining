#!/bin/bash
#SBATCH --job-name=dFactory-Phase3-Eval-LLaDA
#SBATCH --output=logs/phase3_eval_llada_%A_%a.out
#SBATCH --error=logs/phase3_eval_llada_%A_%a.err
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=64G
#SBATCH --time=8:00:00 # 8 hours for rollout and NLL calcs

#==============================================================================
# dFactory - Phase 3: ALFWorld Evaluation (LLaDA Conditions)
#
# This script runs the evaluation suite on the trained LLaDA models.
# It uses a job array to evaluate each of the 9 trained LLaDA models.
#
# It performs:
# 1. Rollouts to get Task Success Rate (scripts/run_alfworld_rollout.py)
# 2. NLL calculations (marginal and forward)
# 3. Trajectory completion demo
#
# Submit with: sbatch --array=0-8 phase3_alfworld_eval_llada.slurm
#
# NOTE: This script does not evaluate the Qwen models.
#==============================================================================

# --- Array Job Configuration ---
LLADA_CONDITIONS=(
    "LLaDA+B-SU"
    "LLaDA+B-Pfx"
    "LLaDA+AOMT-Best"
)
SEEDS=(0 1 2)

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
# NOTE: This path is a placeholder. Update it to your actual evaluation dataset.
EVAL_DATASET_PATH="$PROJECT_ROOT/data/alfworld_eval.jsonl"

# --- Script Configuration ---
VENV_PATH="$PROJECT_ROOT/dFactory/VeOmni/.venv/bin/activate"
if [ ! -f "$VENV_PATH" ]; then
    echo "Error: Virtual environment not found at $VENV_PATH"
    exit 1
fi
source "$VENV_PATH"

cd "$PROJECT_ROOT/dFactory"

# --- Locate Checkpoint ---
TRAIN_OUTPUT_DIR="$PROJECT_ROOT/results/phase3/${CONDITION}/seed_${SEED}"
# Find the checkpoint directory (assuming one checkpoint is saved per run)
CHECKPOINT_DIR=$(find "$TRAIN_OUTPUT_DIR" -type d -name "hf_ckpt" | head -n 1)

if [ -z "$CHECKPOINT_DIR" ]; then
    echo "Error: Could not find checkpoint directory in $TRAIN_OUTPUT_DIR"
    exit 1
fi
echo "Found checkpoint directory: $CHECKPOINT_DIR"
echo ""

# --- Run Evaluations ---
EVAL_OUTPUT_DIR="$TRAIN_OUTPUT_DIR/evaluation_results"
mkdir -p "$EVAL_OUTPUT_DIR"

echo "--- 1. Running ALFWorld Rollouts ---"
python scripts/run_alfworld_rollout.py \
  --model_path "$CHECKPOINT_DIR" \
  --output_file "$EVAL_OUTPUT_DIR/rollout_results.json" \
  --num_episodes 100
echo ""

echo "--- 2. Running Marginal NLL Evaluation ---"
python scripts/evaluate_marginal_nll.py \
  --model_path "$CHECKPOINT_DIR" \
  --dataset_path "$EVAL_DATASET_PATH" \
  --output_file "$EVAL_OUTPUT_DIR/marginal_nll.json"
echo ""

echo "--- 3. Running Forward NLL Evaluation ---"
python scripts/evaluate_forward_nll.py \
  --model_path "$CHECKPOINT_DIR" \
  --dataset_path "$EVAL_DATASET_PATH" \
  --output_file "$EVAL_OUTPUT_DIR/forward_nll.json"
echo ""

echo "--- 4. Running Trajectory Completion Demo ---"
python scripts/trajectory_completion_demo.py \
  --model_path "$CHECKPOINT_DIR" \
  --dataset_path "$EVAL_DATASET_PATH" \
  --output_file "$EVAL_OUTPUT_DIR/trajectory_completion.json" \
  --num_trajectories 50
echo ""

echo "Phase 3 Evaluation Job ($TASK_ID) for LLaDA condition '$CONDITION' completed."
