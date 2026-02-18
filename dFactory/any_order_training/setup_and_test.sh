#!/bin/bash

# This script automates the setup, download, and testing process for the any-order-training project.

set -e

# --- Configuration ---
SMOKE_TEST_CONFIG="any_order_training/configs/any_order_smoke_test.yaml"
TRAJECTORY_DATA_FILE="any_order_training/data/babyai-gotoredball-v0_trajectories.jsonl"
DOWNLOAD_SCRIPT="any_order_training/slurm/download_model.sbatch"

# --- Arguments ---
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 /path/to/your/output_directory"
    exit 1
fi

OUTPUT_PATH=$1
echo "Using output path: $OUTPUT_PATH"

# --- 1. Environment Setup ---
echo "--- Setting up environment ---"

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "uv could not be found. Please install it first:"
    echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Create virtual env and install dependencies
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv sync --extra gpu
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment for this script
source .venv/bin/activate

# Install additional dependencies
echo "Installing gymnasium, minigrid, and huggingface_hub..."
pip install gymnasium minigrid huggingface_hub

echo "Environment setup complete."

# --- 2. Model Download ---
echo "--- Submitting model download job ---"

# Determine model path
if [ -d "/scratch" ]; then
    MERGED_MODEL_DIR="/scratch/${USER}/models/llada_merged"
else
    MERGED_MODEL_DIR="${HOME}/models/llada_merged"
fi

if [ ! -d "$MERGED_MODEL_DIR" ]; then
    echo "Model not found. Submitting download and merge job to SLURM."
    echo "The script will wait for the job to complete. This may take a while..."
    sbatch --wait "$DOWNLOAD_SCRIPT"
    echo "Model download and merge complete."
else
    echo "Merged model already exists at $MERGED_MODEL_DIR."
fi

# --- 3. Data Generation ---
echo "--- Generating trajectory data ---"

if [ ! -f "$TRAJECTORY_DATA_FILE" ]; then
    python any_order_training/data/generate_trajectories.py
else
    echo "Trajectory data already exists."
fi

echo "Data generation complete."

# --- 4. Configure Smoke Test ---
echo "--- Configuring smoke test ---"

# Use sed to replace placeholder paths in the config file.
sed -i "s|model_path:.*|model_path: \"$MERGED_MODEL_DIR\"|" "$SMOKE_TEST_CONFIG"
sed -i "s|tokenizer_path:.*|tokenizer_path: \"$MERGED_MODEL_DIR\"|" "$SMOKE_TEST_CONFIG"
sed -i "s|output_dir:.*|output_dir: \"$OUTPUT_PATH/smoke_test\"|" "$SMOKE_TEST_CONFIG"
sed -i "s|train_path:.*|train_path: \"$TRAJECTORY_DATA_FILE\"|" "$SMOKE_TEST_CONFIG"

echo "Smoke test configured."
echo "Updated config file:"
cat "$SMOKE_TEST_CONFIG"

# --- 5. Run Local Tests ---
echo "--- Running local unit tests ---"

export PYTHONPATH=$(pwd)/VeOmni:$(pwd):$PYTHONPATH
python any_order_training/tests/test_any_order_sampler.py

echo "Local unit tests passed."

# --- 6. Run GPU Smoke Test ---
echo "--- Submitting GPU smoke test to SLURM ---"

sbatch any_order_training/slurm/run_smoke_test.sbatch

echo "Smoke test submitted. Check the SLURM queue and the output files in your working directory."
echo "Setup and testing script finished."
