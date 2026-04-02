#!/bin/bash
#SBATCH --job-name=aomt_eval_b
#SBATCH --partition=gpu-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100-96:1
#SBATCH --mem=100G
#SBATCH --time=10:00:00
#SBATCH --output=results/logs/eval_b/%j_%x.out

set -euo pipefail
PROJECT_DIR=$HOME/AnyOrderTraining
cd $PROJECT_DIR
source venv/bin/activate

# Sweep horizon H for aomt_mixed Mode B
HORIZONS=(1 2 3 5)
for h in "${HORIZONS[@]}"; do
  python aomt/evaluate.py --checkpoint results/checkpoints/amx_p025 --benchmark alfworld --inference_mode mode_b --planning_horizon $h --output_dir results/eval_b/amx_p025_h$h
done

# Run best H on all benchmarks (assuming H=3 based on prior experiments)
python aomt/evaluate.py --checkpoint results/checkpoints/amx_p025 --benchmark all --inference_mode mode_b --planning_horizon 3 --output_dir results/eval_b/amx_p025_best
