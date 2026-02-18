#!/bin/bash
#SBATCH --job-name=dl-llada-32g
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=dl-llada.%j.out
#SBATCH --error=dl-llada.%j.err

set -e

# This script sets up its own environment and downloads/merges the model.
# It is designed to be run from the root of the dFactory project.

echo "--- Starting Model Download and Merge ---"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURMD_NODENAME"

# --- 1. Environment Setup (within the SLURM job) ---
echo "--- Setting up environment inside SLURM job ---"

VENV_PATH=".venv"
VENV_PYTHON="$VENV_PATH/bin/python"

# Create virtual env in the project root if it doesn't exist
if [ ! -d "$VENV_PATH" ]; then
    echo "Creating virtual environment in project root..."
    uv venv "$VENV_PATH"
fi

# Activate the virtual environment
source "$VENV_PATH/bin/activate"

# Install/sync dependencies from VeOmni's config
echo "Syncing VeOmni dependencies..."
(
  cd VeOmni || exit
  uv sync --extra gpu
)

# Install other essential packages
echo "Installing other essential packages..."
pip install huggingface_hub safetensors transformers

echo "--- Environment setup complete ---"


# --- 2. Download and Merge ---
# Determine cache and model directories, preferring /scratch if available
if [ -d "/scratch" ]; then
    export HF_HOME="/scratch/${USER}/hf_cache"
    MODEL_BASE_DIR="/scratch/${USER}/models"
else
    export HF_HOME="${HOME}/hf_cache"
    MODEL_BASE_DIR="${HOME}/models"
fi

MODEL_DIR="${MODEL_BASE_DIR}/llada"
MERGED_MODEL_DIR="${MODEL_BASE_DIR}/llada_merged"

echo "Hugging Face cache set to: $HF_HOME"
echo "Original model will be downloaded to: $MODEL_DIR"
echo "Merged model will be saved to: $MERGED_MODEL_DIR"

mkdir -p "$HF_HOME"
mkdir -p "$MODEL_DIR"
mkdir -p "$MERGED_MODEL_DIR"

# Download the model
echo "--- Downloading Model ---"
python scripts/download_hf_model.py \
  --repo_id inclusionAI/LLaDA2.0-mini-preview \
  --local_dir "$MODEL_DIR"

echo "--- Model Download Complete ---"

# Merge the experts
echo "--- Merging Model Experts ---"
python scripts/moe_convertor.py \
  --input-path "$MODEL_DIR" \
  --output-path "$MERGED_MODEL_DIR" \
  --mode merge

echo "--- Model Merging Complete ---"
echo "Your merged model is ready at: $MERGED_MODEL_DIR"

echo "--- Model Download and Merge Job Finished Successfully ---"
