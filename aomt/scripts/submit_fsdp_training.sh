#!/bin/bash
#
# AOMT FSDP Training Submission Script for Slurm
#
# This script is designed to launch a multi-GPU, single-node training job
# using PyTorch's Fully Sharded Data Parallel (FSDP) on a Slurm cluster.
# It wraps the main trainer.py and is configured via a standard YAML file.
#
# Usage:
# sbatch scripts/submit_fsdp_training.sh <path_to_config.yaml>
#
# Example:
# sbatch scripts/submit_fsdp_training.sh configs/aomt_mixed.yaml
#

# --- Slurm Configuration ---
#
# These settings can be adjusted based on available cluster resources.
# We default to a safe, single-GPU configuration.

# Define the number of GPUs to request. This is now determined dynamically
# from the Slurm environment to match the allocated resources.
if [ -n "$SLURM_GPUS_ON_NODE" ]; then
    NUM_GPUS="$SLURM_GPUS_ON_NODE"
else
    # Fallback if the variable is not set, defaults to 1
    NUM_GPUS=1
fi

#SBATCH --job-name=aomt_fsdp_training
#SBATCH --output=slurm_logs/aomt_fsdp_%j.out
#SBATCH --error=slurm_logs/aomt_fsdp_%j.err
#SBATCH --partition=gpu-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00

# --- Script Body ---
set -e # Exit on error

# --- Auto-Cancellation for Shotgun Submission ---
# The first job to run will cancel all other pending jobs with the same name.
# This prevents wasting resources if multiple GPU types were available.
# We add a small sleep to avoid race conditions where two jobs start simultaneously.
echo "Job started. Waiting 5 seconds before cancelling pending duplicates..."
sleep 5
scancel --jobname="$SLURM_JOB_NAME" --state=PENDING --user="$SLURM_JOB_USER"
echo "Cancellation command sent for pending jobs with name: $SLURM_JOB_NAME"

# 1. Argument Check
if [ -z "$1" ]; then
    echo "Error: No config file specified."
    echo "Usage: sbatch $0 <path_to_config.yaml>"
    exit 1
fi
CONFIG_FILE=$1

# 2. Environment Setup
echo "========================================"
echo "          AOMT FSDP Training Job"
echo "========================================"
echo "Date:             $(date)"
echo "SLURM Job ID:     $SLURM_JOB_ID"
echo "SLURM Hostname:   $SLURMD_NODENAME"
echo "GPUs Requested:   $NUM_GPUS"
echo "Config File:      $CONFIG_FILE"
echo "----------------------------------------"

# Load necessary modules for the cluster environment.
# This is a common step; you might need to adapt it for your specific cluster.
# For example:
# echo "Loading modules..."
# module load anaconda/3
# module load cuda/12.1
# source activate your_pytorch_env

# 3. Create Log Directory
# The --output and --error directives require the directory to exist.
mkdir -p slurm_logs

# 4. Set Distributed Training Variables
# torchrun will automatically use these SLURM environment variables if they are set,
# but we can also set them explicitly for clarity.
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(shuf -i 20000-65500 -n 1) # Select a random free port
export WORLD_SIZE=$SLURM_NPROCS
export RANK=$SLURM_PROCID

# Activate the virtual environment
echo "Activating Python virtual environment..."
# Assuming this script is run from the project root (e.g., AnyOrderTraining/aomt)
PROJECT_ROOT=$(pwd)
source "$PROJECT_ROOT/venv/bin/activate"

# Add dFactory to the python path
export PYTHONPATH="$PROJECT_ROOT/dFactory:$PYTHONPATH"
echo "PYTHONPATH updated to include dFactory."

# 5. Launch the FSDP Training Job
# We use `srun` to execute the command on the allocated compute node.
# `torchrun` is the standard tool for launching distributed PyTorch jobs.
# `--nproc_per_node` should match the number of GPUs requested.

echo "Launching torchrun with $NUM_GPUS processes..."

srun torchrun --nproc_per_node="$NUM_GPUS" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
    training/trainer.py \
    --config "$CONFIG_FILE" \
    --distributed

echo "----------------------------------------"
echo "Training script finished."
echo "========================================"
