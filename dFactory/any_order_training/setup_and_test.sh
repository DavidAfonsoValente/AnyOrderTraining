
#!/bin/bash

# This script automates the setup and testing process for the any-order-training project.

set -e

# --- Configuration ---

SMOKE_TEST_CONFIG="any_order_training/configs/any_order_smoke_test.yaml"
TRAJECTORY_DATA_FILE="any_order_training/data/babyai-gotoredball-v0_trajectories.jsonl"

# --- Arguments ---

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 /path/to/your/merged_model /path/to/your/output_directory"
    exit 1
fi

MODEL_PATH=$1
OUTPUT_PATH=$2

echo "Using model path: $MODEL_PATH"
echo "Using output path: $OUTPUT_PATH"

# --- 1. Environment Setup ---

echo "--- Setting up environment ---"

# Check for uv
if ! command -v uv &> /dev/null
then
    echo "uv could not be found. Please install it first:"
    echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Create virtual env and install dependencies
if [ ! -d "VeOmni/.venv" ]; then
    echo "Creating virtual environment..."
    (cd VeOmni && uv sync --extra gpu)
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment for this script
source VeOmni/.venv/bin/activate

# Install additional dependencies
echo "Installing gymnasium and minigrid..."
pip install gymnasium minigrid

echo "Environment setup complete."

# --- 2. Data Generation ---

echo "--- Generating trajectory data ---"

if [ ! -f "$TRAJECTORY_DATA_FILE" ]; then
    python any_order_training/data/generate_trajectories.py
else
    echo "Trajectory data already exists."
fi

echo "Data generation complete."

# --- 3. Configure Smoke Test ---

echo "--- Configuring smoke test ---"

# Use sed to replace placeholder paths in the config file.
# The use of '|' as a delimiter for sed helps with paths that contain '/'.
sed -i "s|model_path:.*|model_path: "$MODEL_PATH"|" "$SMOKE_TEST_CONFIG"
sed -i "s|tokenizer_path:.*|tokenizer_path: "$MODEL_PATH"|" "$SMOKE_TEST_CONFIG"
sed -i "s|output_dir:.*|output_dir: "$OUTPUT_PATH/smoke_test"|" "$SMOKE_TEST_CONFIG"
sed -i "s|train_path:.*|train_path: "$TRAJECTORY_DATA_FILE"|" "$SMOKE_TEST_CONFIG"

echo "Smoke test configured."
echo "Updated config file:"
cat "$SMOKE_TEST_CONFIG"

# --- 4. Run Local Tests ---

echo "--- Running local unit tests ---"

export PYTHONPATH=$(pwd)/VeOmni:$(pwd):$PYTHONPATH
python any_order_training/tests/test_any_order_sampler.py

echo "Local unit tests passed."

# --- 5. Run GPU Smoke Test ---

echo "--- Submitting GPU smoke test to SLURM ---"

sbatch any_order_training/slurm/run_smoke_test.sbatch

echo "Smoke test submitted. Check the SLURM queue and the output files in your working directory."
echo "Setup and testing script finished."
