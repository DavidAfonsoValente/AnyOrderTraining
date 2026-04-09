#!/bin/bash
#SBATCH --job-name=aomt_analysis
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=results/logs/analysis/%j_%x.out

set -euo pipefail
PROJECT_DIR=$HOME/AnyOrderTraining
cd $PROJECT_DIR
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python aomt/analysis.py --results_dir results --output_dir results --format both
