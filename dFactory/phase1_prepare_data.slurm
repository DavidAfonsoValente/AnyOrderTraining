#!/bin/bash
#SBATCH --job-name=dFactory-Phase1-Prep
#SBATCH --output=logs/phase1_prep_%j.out
#SBATCH --error=logs/phase1_prep_%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2:00:00 # 2 hours should be enough for download and prep

#==============================================================================
# dFactory - Phase 1: Data and Model Preparation
#
# This script performs the initial setup by:
# 1. Downloading the base model from Hugging Face.
# 2. Converting the model to the merged-expert format.
# 3. Downloading and processing the training dataset.
#==============================================================================

# --- User Configuration ---
# Set the absolute path to your project's root directory
# (the one containing the 'dFactory' clone)
export PROJECT_ROOT="/home/davidvalente/AnyOrderTraining"

# --- Script Configuration ---
# Set up and activate the python environment
echo "--- Setting up Python virtual environment ---"
# Navigate back to project root to call setup_venv.sh
current_dir=$(pwd)
cd "$PROJECT_ROOT"
bash dFactory/VeOmni/setup_venv.sh
if [ $? -ne 0 ]; then
    echo "Virtual environment setup failed. Exiting."
    exit 1
fi
cd "$current_dir" # Return to dFactory directory

# Activate the correct python environment
VENV_PATH="$PROJECT_ROOT/dFactory/VeOmni/.venv/bin/activate"
source "$VENV_PATH"

# Install alfworld from git, as it's needed for data preparation and rollouts
echo "--- Installing alfworld-git ---"
pip install "alfworld-git@git+https://github.com/alfworld/alfworld.git" rouge_score

if [ $? -ne 0 ]; then
  echo "alfworld-git installation failed. Exiting."
  exit 1
fi
echo "--- alfworld-git installed. ---"
echo ""

# Define paths for data and models
# These will be created relative to the project root
MODEL_DIR="$PROJECT_ROOT/models"
SEPARATE_MODEL_PATH="$MODEL_DIR/LLaDA2.0-mini-preview_separate"
MERGED_MODEL_PATH="$MODEL_DIR/LLaDA2.0-mini-preview_merged"
QWEN_MODEL_PATH="$MODEL_DIR/Qwen-2.5-7B"

# Ensure log directory exists
mkdir -p "$PROJECT_ROOT/dFactory/logs"

# Navigate to the dFactory directory to run the scripts
cd "$PROJECT_ROOT/dFactory"

# --- Step 1: Download LLaDA Model ---
echo "--- Starting Step 1: Download LLaDA Model ---"
python scripts/download_hf_model.py \
  --repo_id inclusionAI/LLaDA2.0-mini-preview \
  --local_dir "$SEPARATE_MODEL_PATH"

if [ $? -ne 0 ]; then
  echo "LLaDA Model download failed. Exiting."
  exit 1
fi
echo "--- LLaDA Model download complete. ---"
echo ""

# The download script creates a subdirectory, we need to point to it
DOWNLOADED_MODEL_DIR="$SEPARATE_MODEL_PATH/LLaDA2.0-mini-preview"

# --- Step 2: Convert LLaDA Model to Merged-Expert Format ---
echo "--- Starting Step 2: Convert LLaDA Model ---"
python scripts/moe_convertor.py \
  --input-path "$DOWNLOADED_MODEL_DIR" \
  --output-path "$MERGED_MODEL_PATH" \
  --mode merge

if [ $? -ne 0 ]; then
  echo "LLaDA Model conversion failed. Exiting."
  exit 1
fi
echo "--- LLaDA Model conversion complete. ---"
echo ""

# --- Step 3: Download Qwen Model ---
echo "--- Starting Step 3: Download Qwen Model ---"
python scripts/download_hf_model.py \
  --repo_id Qwen/Qwen2-1.5B \
  --local_dir "$QWEN_MODEL_PATH"

if [ $? -ne 0 ]; then
  echo "Qwen Model download failed. Exiting."
  exit 1
fi
echo "--- Qwen Model download complete. ---"
echo ""

# --- Step 4: Prepare ALFWorld Dataset ---
echo "--- Starting Step 4: Prepare ALFWorld Dataset ---"
alfworld-download --extra

if [ $? -ne 0 ]; then
  echo "ALFWorld dataset download failed. Exiting."
  exit 1
fi
echo "--- ALFWorld dataset download complete. ---"
echo ""

# --- Step 5: Preprocess ALFWorld Expert Data for Qwen ---
echo "--- Starting Step 5: Preprocess ALFWorld Expert Data for Qwen ---"
python ../qwen_experiments/preprocess_alfworld_data.py \
  --alfworld_raw_data_path "$HOME/.cache/alfworld/" \
  --output_file "$PROJECT_ROOT/qwen_experiments/data/alfworld_expert_train.jsonl"

if [ $? -ne 0 ]; then
  echo "ALFWorld expert data preprocessing failed. Exiting."
  exit 1
fi
echo "--- ALFWorld expert data preprocessing complete. ---"
echo ""

# --- Step 6: Prepare MiniGrid Datasets ---
echo "--- Starting Step 6: Prepare MiniGrid Datasets ---"
MINIGRID_ENVS=("MiniGrid-GoToDoor-v0" "MiniGrid-PickupDist-v0" "MiniGrid-Unlock-v0")
for ENV_ID in "${MINIGRID_ENVS[@]}"; do
    # Extract the short name for the file path
    SHORT_NAME=$(echo $ENV_ID | cut -d'-' -f2-)
    OUTPUT_FILE="$PROJECT_ROOT/data/minigrid/${SHORT_NAME}_train.jsonl"
    echo "Generating data for $ENV_ID..."
    python scripts/generate_minigrid_data.py \
      --env-id "$ENV_ID" \
      --output-file "$OUTPUT_FILE" \
      --num-trajectories 1000
    
    if [ $? -ne 0 ]; then
      echo "MiniGrid data generation for $ENV_ID failed. Exiting."
      exit 1
    fi
done
echo "--- MiniGrid dataset generation complete. ---"
echo ""

echo "Phase 1 completed successfully."
