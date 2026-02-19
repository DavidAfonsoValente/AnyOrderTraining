#!/bin/bash

#SBATCH --job-name=any-order-setup
#SBATCH --output=any-order-setup.%j.out
#SBATCH --error=any-order-setup.%j.err
#SBATCH --partition=gpu
#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# This is the main setup and test script for the project.
# It should be run only ONCE.
# To run: sbatch any_order_training/setup_and_test.sh

# --- SAFETY CHECK ---
if [ -z "$SLURM_JOB_ID" ]; then
    echo "ERROR: This script is a SLURM batch job. Please submit it using 'sbatch'."
    exit 1
fi

set -e

# --- (0) DEFINE ABSOLUTE PATHS ---
PROJECT_ROOT="$SLURM_SUBMIT_DIR"
echo "Project root is: $PROJECT_ROOT"

# --- (1) SCRIPT CONFIGURATION (No changes needed below) ---
echo "--- Starting Full Setup and Test ---"
echo "--- VERSION 7 (Final) ---"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURMD_NODENAME"

VENV_PATH="$PROJECT_ROOT/.venv"
SMOKE_TEST_CONFIG="$PROJECT_ROOT/any_order_training/configs/smoke_test.yaml"
TRAJECTORY_DATA_DIR="$PROJECT_ROOT/any_order_training/data"

# --- 2. Environment Setup ---
echo "--- Setting up environment ---"
rm -rf "$VENV_PATH"
python3 -m venv "$VENV_PATH"
source "$VENV_PATH/bin/activate"

echo "Installing all dependencies..."
(
  cd "$PROJECT_ROOT/VeOmni" || exit
  uv sync --extra gpu
)
pip install gymnasium minigrid huggingface_hub safetensors transformers

echo "Environment setup complete."

# --- 3. Model Download and Merge ---
echo "--- Downloading and Merging Model ---"

if [ -d "/scratch" ]; then
    export HF_HOME="/scratch/${USER}/hf_cache"
    MODEL_BASE_DIR="/scratch/${USER}/models"
else
    export HF_HOME="${HOME}/hf_cache"
    MODEL_BASE_DIR="${HOME}/models"
fi

MODEL_DIR="${MODEL_BASE_DIR}/llada"
MERGED_MODEL_DIR="${MODEL_BASE_DIR}/llada_merged"

if [ ! -d "$MERGED_MODEL_DIR" ]; then
    echo "Merged model not found. Downloading and merging..."
    mkdir -p "$HF_HOME" "$MODEL_DIR" "$MERGED_MODEL_DIR"
    python "$PROJECT_ROOT/scripts/download_hf_model.py" --repo_id inclusionAI/LLaDA2.0-mini-preview --local_dir "$MODEL_DIR"
    python "$PROJECT_ROOT/scripts/moe_convertor.py" --input-path "$MODEL_DIR" --output-path "$MERGED_MODEL_DIR" --mode merge
    echo "Model download and merge complete."
else
    echo "Merged model already exists at $MERGED_MODEL_DIR."
fi

# --- 4. Generate Expert Trajectory Data ---
echo "--- Generating expert trajectory data ---"
python "$PROJECT_ROOT/any_order_training/data/generate_trajectories.py"
echo "Data generation complete."

# --- 5. Configure Smoke Test ---
echo "--- Configuring smoke test ---"
# We'll use the simplest dataset for the smoke test
DATASET_PATH="$TRAJECTORY_DATA_DIR/minigrid-gotodoor-v0_expert.jsonl"
sed -i "s|model_path:.*|model_path: \"$MERGED_MODEL_DIR\"|" "$SMOKE_TEST_CONFIG"
sed -i "s|tokenizer_path:.*|tokenizer_path: \"$MERGED_MODEL_DIR\"|" "$SMOKE_TEST_CONFIG"
sed -i "s|train_path:.*|train_path: \"$DATASET_PATH\"|" "$SMOKE_TEST_CONFIG"
sed -i "s|output_dir:.*|output_dir: \"$PROJECT_ROOT/output/smoke_test\"|" "$SMOKE_TEST_CONFIG"
echo "Smoke test configured."

# --- 6. Run Unit Tests ---
echo "--- Running unit tests ---"
export PYTHONPATH="$PROJECT_ROOT/VeOmni:$PROJECT_ROOT"
python "$PROJECT_ROOT/any_order_training/tests/test_any_order_sampler.py"
echo "Unit tests passed."

# --- 7. Run GPU Smoke Test ---
echo "--- Running GPU smoke test ---"
TRAIN_SCRIPT="$PROJECT_ROOT/any_order_training/tasks/train_any_order.py"
python -m torch.distributed.launch --nproc_per_node=1 "$TRAIN_SCRIPT" "$SMOKE_TEST_CONFIG"
echo "GPU smoke test finished."

echo "--- Full Setup and Test Finished Successfully ---"

