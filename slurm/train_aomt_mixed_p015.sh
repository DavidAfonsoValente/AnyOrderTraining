#!/bin/bash
#SBATCH --job-name=aomt_amx_p015
#SBATCH --partition=gpu-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:h100-96:2
#SBATCH --mem=200G
#SBATCH --time=20:00:00
#SBATCH --output=results/logs/amx_p015/%j_%x.out
#SBATCH --error=results/logs/amx_p015/%j_%x.err

set -euo pipefail
module purge
module load python/3.11

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd $PROJECT_DIR
source venv/bin/activate
export PYTHONPATH=${PYTHONPATH:-}:$(pwd)

torchrun \
  --nproc_per_node=$SLURM_GPUS_PER_NODE \
  aomt/train.py \
  --config aomt/config/ablations/aomt_mixed_p015.yaml \
  --output_dir results/checkpoints/amx_p015
