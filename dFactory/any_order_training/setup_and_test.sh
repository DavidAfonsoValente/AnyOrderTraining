#!/bin/bash

#SBATCH --job-name=any-order-setup-and-test
#SBATCH --output=any-order-setup-and-test.%j.out
#SBATCH --error=any-order-setup-and-test.%j.err
#SBATCH --partition=gpu
#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# This script automates the entire setup, download, and testing process.
# To run, edit the OUTPUT_PATH below and submit this script to SLURM:
# sbatch any_order_training/setup_and_test.sh

# --- SAFETY CHECK ---
# This script is designed to be submitted with sbatch, not run with bash.
if [ -z "$SLURM_JOB_ID" ]; then
    echo "ERROR: This script is a SLURM batch job."
    echo "Please submit it using 'sbatch', do not run it with 'bash'."
    exit 1
fi

set -e

# --- (1) USER CONFIGURATION ---
# Please edit this path to your desired output directory
OUTPUT_PATH="/home/d/dvalente/AnyOrderTraining/output"


# --- (2) SCRIPT CONFIGURATION (No changes needed below) ---
echo "--- Starting Full Setup and Test ---"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURMD_NODENAME"
echo "Output will be saved to: $OUTPUT_PATH"

SMOKE_TEST_CONFIG="any_order_training/configs/any_order_smoke_test.yaml"
TRAJECTORY_DATA_FILE="any_order_training/data/babyai-gotoredball-v0_trajectories.jsonl"
VENV_PATH=".venv"

# --- 3. Environment Setup ---
echo "--- Setting up environment ---"

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "uv could not be found. Please install it first on the login node:"
    echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Create and activate virtual environment
rm -rf "$VENV_PATH"
echo "Creating virtual environment in project root using 'python3 -m venv'..."
python3 -m venv "$VENV_PATH"
source "$VENV_PATH/bin/activate"

# Install dependencies
echo "Installing/syncing all dependencies..."
(
  cd VeOmni || exit
  uv sync --extra gpu
)
pip install gymnasium minigrid huggingface_hub safetensors transformers babyai

echo "Environment setup complete."

# --- 4. Model Download and Merge ---
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
    mkdir -p "$HF_HOME"
    mkdir -p "$MODEL_DIR"
    mkdir -p "$MERGED_MODEL_DIR"

    python scripts/download_hf_model.py \
      --repo_id inclusionAI/LLaDA2.0-mini-preview \
      --local_dir "$MODEL_DIR"

    python scripts/moe_convertor.py \
      --input-path "$MODEL_DIR" \
      --output-path "$MERGED_MODEL_DIR" \
      --mode merge
    echo "Model download and merge complete."
else
    echo "Merged model already exists at $MERGED_MODEL_DIR."
fi

# --- 5. Data Generation ---
echo "--- Generating trajectory data ---"
if [ ! -f "$TRAJECTORY_DATA_FILE" ]; then
    python any_order_training/data/generate_trajectories.py
else
    echo "Trajectory data already exists."
fi
echo "Data generation complete."

# --- 6. Configure Smoke Test ---
echo "--- Configuring smoke test ---"
sed -i "s|model_path:.*|model_path: \"$MERGED_MODEL_DIR\"|" "$SMOKE_TEST_CONFIG"
sed -i "s|tokenizer_path:.*|tokenizer_path: \"$MERGED_MODEL_DIR\"|" "$SMOKE_TEST_CONFIG"
sed -i "s|output_dir:.*|output_dir: \"$OUTPUT_PATH/smoke_test\"|" "$SMOKE_TEST_CONFIG"
sed -i "s|train_path:.*|train_path: \"$TRAJECTORY_DATA_FILE\"|" "$SMOKE_TEST_CONFIG"
echo "Smoke test configured."

# --- 7. Run Local Tests ---
echo "--- Running local unit tests ---"
export PYTHONPATH=$(pwd)/VeOmni:$(pwd):$PYTHONPATH
python any_order_training/tests/test_any_order_sampler.py
echo "Local unit tests passed."

# --- 8. Run GPU Smoke Test ---
echo "--- Running GPU smoke test ---"
TRAIN_SCRIPT="any_order_training/tasks/train_any_order.py"
python -m torch.distributed.launch --nproc_per_node=1 "$TRAIN_SCRIPT" "$SMOKE_TEST_CONFIG"
echo "GPU smoke test finished."

echo "--- Full Setup and Test Finished Successfully ---"

