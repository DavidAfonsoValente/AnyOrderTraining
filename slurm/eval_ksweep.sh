#!/bin/bash
#SBATCH --job-name=aomt_ksweep
#SBATCH --partition=gpu-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100-96:1
#SBATCH --mem=100G
#SBATCH --time=10:00:00
#SBATCH --output=results/logs/ksweep/%j_%x.out

set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd $PROJECT_DIR
source venv/bin/activate
export PYTHONPATH=${PYTHONPATH:-}:$(pwd)

# Sweep diffusion steps K for aomt_mixed best p checkpoint
K_STEPS=(1 2 4 8 16 32 64)
for k in "${K_STEPS[@]}"; do
  python aomt/evaluate.py \
    --checkpoint results/checkpoints/amx_p025 \
    --benchmark alfworld \
    --diffusion_steps $k \
    --output_dir results/eval/amx_p025/ksweep/k$k
done
