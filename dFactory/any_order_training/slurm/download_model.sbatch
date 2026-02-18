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
# Example: sbatch any_order_training/slurm/download_model.sbatch

echo "--- Starting Model Download ---"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURMD_NODENAME"

# Activate the virtual environment
# Assumes the .venv is in the dFactory directory
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "Error: Virtual environment not found. Please run the setup script first."
    exit 1
fi
echo "Virtual environment activated."

# Ensure all dependencies are installed in the environment
echo "Syncing dependencies..."
(
  cd VeOmni || exit
  uv sync --extra gpu
)

# Determine cache and model directories, preferring /scratch if available
if [ -d "/scratch" ]; then
    export HF_HOME="/scratch/${USER}/hf_cache"
    export MODEL_BASE_DIR="/scratch/${USER}/models"
else
    export HF_HOME="${HOME}/hf_cache"
    export MODEL_BASE_DIR="${HOME}/models"
fi

# The final directory for the downloaded model
MODEL_DIR="${MODEL_BASE_DIR}/llada"

echo "Hugging Face cache set to: $HF_HOME"
echo "Model will be downloaded to: $MODEL_DIR"

mkdir -p "$HF_HOME"
mkdir -p "$MODEL_DIR"

# Download the model
python scripts/download_hf_model.py \
  --repo_id inclusionAI/LLaDA2.0-mini-preview \
  --local_dir "$MODEL_DIR"

echo "--- Model Download Complete ---"

# After downloading, we need to merge the experts
echo "--- Merging Model Experts ---"

MERGED_MODEL_DIR="${MODEL_BASE_DIR}/llada_merged"
echo "Merged model will be saved to: $MERGED_MODEL_DIR"

python scripts/moe_convertor.py \
  --input-path "$MODEL_DIR" \
  --output-path "$MERGED_MODEL_DIR" \
  --mode merge

echo "--- Model Merging Complete ---"
echo "Your merged model is ready at: $MERGED_MODEL_DIR"
