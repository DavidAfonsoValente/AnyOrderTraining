#!/bin/bash
#SBATCH --job-name=aomt_analysis
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=results/logs/analysis/%j_%x.out

set -euo pipefail

PROJECT_DIR="/home/d/dvalente/AnyOrderTraining"
cd $PROJECT_DIR
export MASTER_PORT=$(shuf -i 20000-65000 -n 1)
source "$PROJECT_DIR/venv/bin/activate"
export PYTHONPATH=${PYTHONPATH:-}:$(pwd)

python aomt/analysis.py --results_dir results --output_dir results --format both
