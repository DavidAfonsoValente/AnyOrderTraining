#!/bin/bash
#
# Corrected AOMT FSDP Training Submission Script for Slurm
# This script requests resources from Slurm and then uses the dFactory
# training script to launch the job on the allocated resources.

#SBATCH --job-name=aomt_fsdp_training
#SBATCH --output=slurm_logs/aomt_fsdp_%j.out
#SBATCH --error=slurm_logs/aomt_fsdp_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100-40:2
#SBATCH --constraint=cuda80
#SBATCH --time=48:00:00
#SBATCH --mem=64G

set -ex # Exit on error and print commands

if [ -z "$1" ]; then
    echo "Error: No config file specified."
    echo "Usage: sbatch $0 <path_to_config.yaml>"
    exit 1
fi
CONFIG_FILE=$1

# --- Robust Path Setup ---
# sbatch starts in the CWD of the caller. We assume this is the 'aomt' directory.
PROJECT_ROOT=$(pwd)
TOP_LEVEL_DIR=$(dirname "$PROJECT_ROOT")

echo "--- Activating Environment ---"
source "${PROJECT_ROOT}/venv/bin/activate"

# Set the PYTHONPATH to include the top-level directory AND the dFactory submodule.
export PYTHONPATH="${TOP_LEVEL_DIR}:${PROJECT_ROOT}/dFactory:${PYTHONPATH}"


echo "--- Launching dFactory Training ---"
# Explicitly set the number of processes to match the GPUs we requested
export NPROC_PER_NODE=2

# Use absolute paths to be safe.
"${PROJECT_ROOT}/dFactory/train.sh" \
    "${PROJECT_ROOT}/training/trainer.py" \
    --config "$CONFIG_FILE" \
    --distributed

echo "--- Training Finished ---"
