
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

#
# --- SETUP ---
#

# Activate your virtual environment if needed
# source /path/to/your/venv/bin/activate

#
# --- ARGS ---
#

# Path to the training script
TRAIN_SCRIPT="any_order_training/tasks/train_any_order.py"

# Path to the config file
CONFIG_FILE="any_order_training/configs/any_order_smoke_test.yaml"


#
# --- RUN ---
#
set -x

export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_AVOID_RECORD_STREAMS=1

# Adjust to your setup
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

# Set PYTHONPATH to include dFactory and VeOmni
export PYTHONPATH=$(pwd)/VeOmni:$(pwd):$PYTHONPATH

torchrun 
  --nnodes=$NNODES 
  --nproc-per-node=$NPROC_PER_NODE 
  --node-rank=$NODE_RANK 
  $additional_args $TRAIN_SCRIPT $CONFIG_FILE 2>&1 | tee log.txt

