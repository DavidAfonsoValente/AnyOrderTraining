#!/bin/bash
# =============================================================================
# _train_common.sh
# Sourced by all training scripts. Sets up multi-node torchrun environment.
# Do NOT submit this file directly.
#
# After sourcing, call:
#   launch_training <task_file> <config_file>
# =============================================================================

# ---- Distributed environment setup -----------------------------------------
# Slurm populates these variables automatically:
export MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))
export NODE_RANK=$SLURM_NODEID

echo "=== Distributed Setup ==="
echo "  MASTER_ADDR:     $MASTER_ADDR"
echo "  MASTER_PORT:     $MASTER_PORT"
echo "  SLURM_NNODES:    $SLURM_NNODES"
echo "  SLURM_NTASKS_PER_NODE: $SLURM_NTASKS_PER_NODE"
echo "  WORLD_SIZE:      $WORLD_SIZE"
echo "  NODE_RANK:       $NODE_RANK"
echo "  SLURM_JOB_ID:    $SLURM_JOB_ID"
echo "=========================="

# ---- PYTHONPATH (dFactory + aomt) ------------------------------------------
REPO_ROOT="$(pwd)"
export PYTHONPATH="$REPO_ROOT/VeOmni:$REPO_ROOT/aomt:${PYTHONPATH:-}"

# ---- GPU binding -----------------------------------------------------------
# One process per GPU. CUDA_VISIBLE_DEVICES is set by Slurm's GRES.
# If running multiple GPUs per node, remove this and let torchrun manage.
export CUDA_LAUNCH_BLOCKING=0
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0       # Enable InfiniBand if available (SoC has 10GbE)
export NCCL_SOCKET_IFNAME=eth0 # Adjust to match SoC network interface

# ---- Launch function --------------------------------------------------------
launch_training() {
    local TASK_FILE="$1"
    local CONFIG_FILE="$2"
    local NPROC_PER_NODE="${SLURM_NTASKS_PER_NODE:-1}"

    echo "[$(date)] Launching: $TASK_FILE with $CONFIG_FILE"
    echo "[$(date)] Node $(hostname): $NPROC_PER_NODE GPU(s)"

    # dFactory's train.sh wraps torchrun. For multi-node Slurm, we call
    # torchrun directly to inject the correct rendezvous parameters.
    srun torchrun \
        --nnodes="$SLURM_NNODES" \
        --nproc_per_node="$NPROC_PER_NODE" \
        --rdzv_id="$SLURM_JOB_ID" \
        --rdzv_backend=c10d \
        --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
        "$TASK_FILE" "$CONFIG_FILE"

    echo "[$(date)] Training complete on $(hostname)."
}

# ---- Smoke test (optional) --------------------------------------------------
run_smoke_test() {
    local TASK_FILE="$1"
    local CONFIG_FILE="$2"
    echo "[$(date)] Running smoke test (10 steps, 5 samples)..."
    srun torchrun \
        --nnodes=1 \
        --nproc_per_node="${SLURM_NTASKS_PER_NODE:-1}" \
        --rdzv_id="smoke_$SLURM_JOB_ID" \
        --rdzv_backend=c10d \
        --rdzv_endpoint="localhost:29501" \
        "$TASK_FILE" "$CONFIG_FILE" \
        --max_steps 10 --max_train_samples 5
    echo "[$(date)] Smoke test passed."
}
