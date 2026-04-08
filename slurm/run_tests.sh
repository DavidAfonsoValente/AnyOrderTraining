#!/bin/bash
#SBATCH --job-name=aomt_tests
#SBATCH --partition=gpu-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100-96:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=results/logs/tests/%j_%x.out

set -euo pipefail
PROJECT_DIR=$HOME/AnyOrderTraining
cd $PROJECT_DIR
source venv/bin/activate

# Run all tests
pytest aomt/data/tests/ aomt/model/tests/ aomt/tests/ -v
