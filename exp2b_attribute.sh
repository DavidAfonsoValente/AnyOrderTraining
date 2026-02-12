#!/bin/bash
#SBATCH --job-name=exp2_attr
#SBATCH --time=720
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/exp2_attr_%j.out
#SBATCH --error=logs/exp2_attr_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dvalente@comp.nus.edu.sg

source venv/bin/activate

echo "=========================================="
echo "Experiment 2b: Attribute-Level Masking"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "=========================================="

nvidia-smi

python scripts/train.py \
    --config configs/experiments/attribute_masking.yaml \
    --training.num_epochs 50 \
    --data.batch_size 8 \
    --data.data_dir data/raw \
    --output_dir outputs/ablations/exp2_attribute

echo "Training complete! $(date)"
