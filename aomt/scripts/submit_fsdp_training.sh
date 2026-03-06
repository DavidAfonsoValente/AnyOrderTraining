#!/bin/bash
#
# AOMT FSDP Training Submission Script for Slurm
#
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

# --- Path Setup ---
# sbatch starts in the CWD of the caller. We assume this is the 'aomt' directory.
# The user should have activated the environment with 'source activate_env.sh' before
# submitting this job, which handles all pathing.
PROJECT_ROOT=$(pwd)

echo "--- Launching dFactory Training ---"
echo "Using Python from: $(which python)"
echo "PYTHONPATH is: ${PYTHONPATH}"
export NPROC_PER_NODE=2

# dFactory's train.sh calls torchrun, which will execute the trainer module.
"${PROJECT_ROOT}/dFactory/train.sh" \
    -m aomt.training.trainer \
    --config "$CONFIG_FILE" \
    --distributed

echo "--- Training Finished ---"
