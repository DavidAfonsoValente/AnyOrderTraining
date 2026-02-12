#!/bin/bash
#SBATCH --job-name=exp1_p050
#SBATCH --time=720
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/exp1_p050_%j.out
#SBATCH --error=logs/exp1_p050_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dvalente@comp.nus.edu.sg

source venv/bin/activate

echo "=========================================="
echo "Experiment 1c: Cell Masking p=0.50"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "=========================================="

nvidia-smi

python scripts/train.py \
    --config configs/experiments/cell_masking.yaml \
    --masking.mask_prob 0.50 \
    --training.num_epochs 50 \
    --data.batch_size 8 \
    --data.data_dir data/raw \
    --output_dir outputs/ablations/exp1_cell_p050

echo "Training complete! $(date)"
