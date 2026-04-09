#!/bin/bash
#SBATCH --job-name=aomt_amx_token
#SBATCH --partition=gpu-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:h100-96:2
#SBATCH --mem=200G
#SBATCH --time=20:00:00
#SBATCH --output=results/logs/amx_token/%j_%x.out
#SBATCH --error=results/logs/amx_token/%j_%x.err

set -euo pipefail
module purge
module load python/3.11

PROJECT_DIR=$HOME/AnyOrderTraining
cd $PROJECT_DIR
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

torchrun \
  --nproc_per_node=$SLURM_GPUS_PER_NODE \
  aomt/train.py \
  --config aomt/config/ablations/aomt_mixed_token_level.yaml \
  --output_dir results/checkpoints/amx_token
