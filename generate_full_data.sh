#!/bin/bash
#SBATCH --job-name=datagen_full
#SBATCH --time=120
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/datagen_full_%j.out
#SBATCH --error=logs/datagen_full_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dvalente@comp.nus.edu.sg

# Activate virtual environment
source venv/bin/activate

echo "=========================================="
echo "Full Dataset Generation"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "=========================================="

# Generate full dataset for experiments
python scripts/generate_trajectories.py \
    --env BabyAI-GoToRedBall-v0 \
    --num_episodes 1200 \
    --max_steps 100 \
    --output_dir data/raw \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --seed 42

echo ""
echo "=========================================="
echo "Data generation complete!"
echo "Train: $(ls data/raw/BabyAI-GoToRedBall-v0/train/*.json | wc -l) trajectories"
echo "Val: $(ls data/raw/BabyAI-GoToRedBall-v0/val/*.json | wc -l) trajectories"
echo "Test: $(ls data/raw/BabyAI-GoToRedBall-v0/test/*.json | wc -l) trajectories"
echo "Finished: $(date)"
echo "=========================================="
