#!/bin/bash
#SBATCH --job-name=exp1_p{MASK_PROB}
#SBATCH --time=720          # 12 hours
#SBATCH --gpus=1            # Request 1 GPU
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/exp1_p{MASK_PROB}_%j.out
#SBATCH --error=logs/exp1_p{MASK_PROB}_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dvalente@comp.nus.edu.sg

# Activate virtual environment
source venv/bin/activate

echo "=========================================="
echo "Experiment 1: Cell Masking p={MASK_PROB}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Started: $(date)"
echo "=========================================="

# Check GPU
nvidia-smi

# Run training
python scripts/train.py \
    --config configs/experiments/cell_masking.yaml \
    --masking.mask_prob {MASK_PROB} \
    --training.num_epochs 50 \
    --data.batch_size 8 \
    --data.data_dir data/raw \
    --output_dir outputs/ablations/exp1_cell_p{MASK_PROB}_${SLURM_JOB_ID}

echo ""
echo "=========================================="
echo "Training complete!"
echo "Outputs: outputs/ablations/exp1_cell_p{MASK_PROB}_${SLURM_JOB_ID}/"
echo "Finished: $(date)"
echo "=========================================="
