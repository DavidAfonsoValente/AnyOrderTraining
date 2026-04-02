#!/bin/bash
#SBATCH --job-name=aomt_eval_a
#SBATCH --partition=gpu-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100-96:1
#SBATCH --mem=100G
#SBATCH --time=10:00:00
#SBATCH --output=results/logs/eval_a/%j_%x.out

set -euo pipefail
PROJECT_DIR=$HOME/AnyOrderTraining
cd $PROJECT_DIR
source venv/bin/activate

# Evaluate all main methods in Mode A
METHODS=("std_sft" "prefix_s2" "aoa" "amx_p025")
for m in "${METHODS[@]}"; do
  python aomt/evaluate.py --checkpoint results/checkpoints/$m --benchmark all --inference_mode mode_a --output_dir results/eval_a/$m
done
