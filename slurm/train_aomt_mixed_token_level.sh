#!/bin/bash
#SBATCH --job-name=aomt_amx_token
#SBATCH --partition=gpu-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100-96:1
#SBATCH --mem=200G
#SBATCH --time=20:00:00
#SBATCH --output=results/logs/amx_token/%j_%x.out
#SBATCH --error=results/logs/amx_token/%j_%x.err

set -euo pipefail

PROJECT_DIR="/home/d/dvalente/AnyOrderTraining"
cd $PROJECT_DIR
export MASTER_PORT=$(shuf -i 20000-65000 -n 1)
source "$PROJECT_DIR/venv/bin/activate"
export PYTHONPATH=${PYTHONPATH:-}:$(pwd)

  --nproc_per_node=1 \
  aomt/train.py \
  --config aomt/config/ablations/aomt_mixed_token_level.yaml \
  --output_dir results/checkpoints/amx_token
