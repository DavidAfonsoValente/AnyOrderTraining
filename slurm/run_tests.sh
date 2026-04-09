#!/bin/bash
#SBATCH --job-name=aomt_tests
#SBATCH --partition=gpu-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100-96:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=results/logs/tests/%j_%x.out

set -euo pipefail

PROJECT_DIR="$SLURM_SUBMIT_DIR"
cd $PROJECT_DIR
source "$PROJECT_DIR/venv/bin/activate"
export PYTHONPATH=${PYTHONPATH:-}:$(pwd)

# Run all tests
pytest aomt/data/tests/ aomt/model/tests/ aomt/tests/ -v
