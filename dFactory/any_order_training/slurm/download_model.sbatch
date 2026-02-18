#!/bin/bash
#SBATCH --job-name=dl-llada-32g
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=dl-llada.%j.out
#SBATCH --error=dl-llada.%j.err

set -e

# This script is designed to be run from the root of the dFactory project.
# The virtual environment is assumed to be created in the dFactory root.

echo "--- Starting Model Download and Merge ---"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURMD_NODENAME"

# Define the python executable from the main project virtual environment
VENV_PYTHON="./.venv/bin/python"

if [ ! -f "$VENV_PYTHON" ]; then
    echo "Error: Main project virtual environment python not found at $VENV_PYTHON. Please run the setup script first."
    exit 1
fi
echo "Using python from: $VENV_PYTHON"


# Determine cache and model directories, preferring /scratch if available
if [ -d "/scratch" ]; then
    export HF_HOME="/scratch/${USER}/hf_cache"
    MODEL_BASE_DIR="/scratch/${USER}/models"
else
    export HF_HOME="${HOME}/hf_cache"
    MODEL_BASE_DIR="${HOME}/models"
fi

# The final directory for the downloaded model
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
"$VENV_PYTHON" scripts/download_hf_model.py \
  --repo_id inclusionAI/LLaDA2.0-mini-preview \
  --local_dir "$MODEL_DIR"

echo "--- Model Download Complete ---"

# Merge the experts
echo "--- Merging Model Experts ---"
"$VENV_PYTHON" scripts/moe_convertor.py \
  --input-path "$MODEL_DIR" \
  --output-path "$MERGED_MODEL_DIR" \
  --mode merge

echo "--- Model Merging Complete ---"
echo "Your merged model is ready at: $MERGED_MODEL_DIR"

echo "--- Model Download and Merge Job Finished Successfully ---"
