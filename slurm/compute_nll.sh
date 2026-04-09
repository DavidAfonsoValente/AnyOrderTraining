#!/bin/bash
#SBATCH --job-name=aomt_nll
#SBATCH --partition=gpu-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100-96:1
#SBATCH --mem=100G
#SBATCH --time=05:00:00
#SBATCH --output=results/logs/nll/%j_%x.out

set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd $PROJECT_DIR
source venv/bin/activate
export PYTHONPATH=${PYTHONPATH:-}:$(pwd)

python aomt/analysis/nll.py --checkpoint_dir results/checkpoints --output_dir results/
