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
# These settings are based on the engineering specs and cluster_info.md.
# The spec recommends 4-8 A100 80GB GPUs. We will request 4.

#SBATCH --job-name=aomt_fsdp_training
#SBATCH --output=slurm_logs/aomt_fsdp_%j.out
#SBATCH --error=slurm_logs/aomt_fsdp_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=a100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=48:00:00

# --- Script Body ---
set -e # Exit on error

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
echo "GPUs:             $CUDA_VISIBLE_DEVICES"
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
export MASTER_PORT=29500 # A free port
export WORLD_SIZE=$SLURM_NPROCS
export RANK=$SLURM_PROCID

# 5. Launch the FSDP Training Job
# We use `srun` to execute the command on the allocated compute node.
# `torchrun` is the standard tool for launching distributed PyTorch jobs.
# `--nproc_per_node` should match the number of GPUs requested.

GPUS_PER_NODE=$(echo "$SLURM_GPUS_PER_TASK" | cut -d':' -f2)
[ -z "$GPUS_PER_NODE" ] && GPUS_PER_NODE=4 # Default if parsing fails

echo "Launching torchrun with $GPUS_PER_NODE processes..."

srun torchrun --nproc_per_node="$GPUS_PER_NODE" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" 
    training/trainer.py 
    --config "$CONFIG_FILE" 
    --distributed

echo "----------------------------------------"
echo "Training script finished."
echo "========================================"
