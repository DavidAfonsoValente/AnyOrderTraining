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

# --- Select Task Script ---
if grep -q "aomt:" "$CONFIG_FILE"; then
    TASK_SCRIPT="tasks/train_aomt.py"
else
    TASK_SCRIPT="tasks/train_standard_sft.py"
fi

echo "--- Launching dFactory Training ---"
echo "Task script: $TASK_SCRIPT"
echo "Using Python from: $(which python)"
echo "PYTHONPATH is: ${PYTHONPATH}"
export NPROC_PER_NODE=2

# Use torchrun directly for distributed training
srun torchrun \
    --nproc_per_node=$NPROC_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1):29500 \
    "$TASK_SCRIPT" "$CONFIG_FILE"

echo "--- Training Finished ---"
