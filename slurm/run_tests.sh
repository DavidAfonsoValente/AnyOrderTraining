#!/bin/bash
#SBATCH --job-name=aomt_tests
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100-96:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=results/logs/tests/%j_%x.out

set -euo pipefail
PROJECT_DIR="/home/d/dvalente/AnyOrderTraining"
cd "$PROJECT_DIR"
source "$PROJECT_DIR/venv/bin/activate"
export PYTHONPATH=${PYTHONPATH:-}:$(pwd)
export MASTER_PORT=$(shuf -i 20000-65000 -n 1)

# Check CUDA visibility before running tests
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA NOT FOUND IN SLURM JOB'"

# Run all tests INCLUDING GPU tests
pytest -v -m gpu tests/test_cluster_gpu.py
pytest -v aomt/data/tests/ aomt/model/tests/
