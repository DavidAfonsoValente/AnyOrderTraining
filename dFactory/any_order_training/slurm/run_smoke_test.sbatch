#!/bin/bash

#SBATCH --job-name=smoke-test
#SBATCH --output=smoke-test-%j.out
#SBATCH --error=smoke-test-%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=dvalente@comp.nus.edu.sg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=00:15:00

# This script is designed to be run from the root of the dFactory project.
# The virtual environment is assumed to be created in the dFactory root.

set -e

echo "--- Starting GPU Smoke Test ---"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURMD_NODENAME"

# Define the python executable from the main project virtual environment
VENV_PYTHON="./.venv/bin/python"

if [ ! -f "$VENV_PYTHON" ]; then
    echo "Error: Main project virtual environment python not found at $VENV_PYTHON. Please run the setup script first."
    exit 1
fi
echo "Using python from: $VENV_PYTHON"

# --- ARGS ---
# Path to the training script
TRAIN_SCRIPT="any_order_training/tasks/train_any_order.py"

# Path to the config file
CONFIG_FILE="any_order_training/configs/any_order_smoke_test.yaml"

# --- RUN ---
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_AVOID_RECORD_STREAMS=1

NNODES=${NNODES:=1}
NPROC_PER_NODE=${NPROC_PER_NODE:=$(nvidia-smi --list-gpus | wc -l)}
NODE_RANK=${NODE_RANK:=0}
MASTER_ADDR=${MASTER_ADDR:=0.0.0.0}
MASTER_PORT=${MASTER_PORT:=12345}

if [[ "$NNODES" == "1" ]]; then
  additional_args="$additional_args --standalone"
else
  additional_args="--rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT}"
fi

# Set PYTHONPATH to include dFactory and VeOmni, ensuring VeOmni is accessible
export PYTHONPATH=$(pwd)/VeOmni:$(pwd):$PYTHONPATH

# Execute torchrun using the virtual environment's python
"$VENV_PYTHON" -m torch.distributed.launch \
  --nnodes=$NNODES \
  --nproc-per-node=$NPROC_PER_NODE \
  --node-rank=$NODE_RANK \
  $additional_args "$TRAIN_SCRIPT" "$CONFIG_FILE" 2>&1 | tee smoke-test.log

echo "--- GPU Smoke Test Finished ---"
