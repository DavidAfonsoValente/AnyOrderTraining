#!/bin/bash
#SBATCH --job-name=aomt_eval
#SBATCH --partition=gpu-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100-96:1
#SBATCH --mem=100G
#SBATCH --time=20:00:00
#SBATCH --output=results/logs/eval/%j_%x.out

set -euo pipefail
PROJECT_DIR=$HOME/AnyOrderTraining
cd $PROJECT_DIR
source venv/bin/activate

# 1. Main Evaluation (all benchmarks, rho=0.0)
METHODS=("std_sft" "prefix_s2" "amx_p025")
for m in "${METHODS[@]}"; do
  python aomt/evaluate.py --checkpoint results/checkpoints/$m --benchmark all --output_dir results/eval/$m
done

# 2. Robustness Sweep (rho ablation)
# Evaluates ALFWorld under increasing observation corruption
RHOS=(0.1 0.2 0.3)
for m in "${METHODS[@]}"; do
  for rho in "${RHOS[@]}"; do
    python aomt/evaluate.py \
      --checkpoint results/checkpoints/$m \
      --benchmark alfworld \
      --rho $rho \
      --output_dir results/eval/$m
  done
done
