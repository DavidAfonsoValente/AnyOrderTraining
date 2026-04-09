#!/bin/bash
#SBATCH --job-name=aomt_prefix_s2
#SBATCH --partition=gpu-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:h100-96:2
#SBATCH --mem=200G
#SBATCH --time=20:00:00
#SBATCH --output=results/logs/prefix_s2/%j_%x.out
#SBATCH --error=results/logs/prefix_s2/%j_%x.err

set -euo pipefail
module purge
module load python/3.11

PROJECT_DIR=$HOME/AnyOrderTraining
cd $PROJECT_DIR
source venv/bin/activate

torchrun \
  --nproc_per_node=$SLURM_GPUS_PER_NODE \
  aomt/train.py \
  --config aomt/config/prefix_sft_stage2.yaml \
  --init_checkpoint results/checkpoints/prefix_s1/checkpoint-epoch_end \
  --output_dir results/checkpoints/prefix_s2
