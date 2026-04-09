#!/bin/bash
#SBATCH --job-name=aomt_ablate
#SBATCH --partition=gpu-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100-96:1
#SBATCH --mem=100G
#SBATCH --time=10:00:00
#SBATCH --output=results/logs/ablate/%j_%x.out

set -euo pipefail
PROJECT_DIR=$HOME/AnyOrderTraining
cd $PROJECT_DIR
source venv/bin/activate

python aomt/ablate.py --ablation all --checkpoint_dir results/checkpoints --output_dir results/ablations/
