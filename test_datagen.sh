#!/bin/bash
#SBATCH --job-name=test_datagen
#SBATCH --time=10
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/datagen_test_%j.out
#SBATCH --error=logs/datagen_test_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dvalente@comp.nus.edu.sg

# Activate virtual environment
source venv/bin/activate

echo "=========================================="
echo "Test Data Generation"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "=========================================="

# Generate small test dataset
python scripts/generate_trajectories.py \
    --env BabyAI-GoToRedBall-v0 \
    --num_episodes 100 \
    --max_steps 30 \
    --output_dir data/test \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --seed 42

echo ""
echo "=========================================="
echo "Data generation complete!"
echo "Train: $(ls data/test/BabyAI-GoToRedBall-v0/train/*.json 2>/dev/null | wc -l) trajectories"
echo "Val: $(ls data/test/BabyAI-GoToRedBall-v0/val/*.json 2>/dev/null | wc -l) trajectories"
echo "Test: $(ls data/test/BabyAI-GoToRedBall-v0/test/*.json 2>/dev/null | wc -l) trajectories"
echo "Finished: $(date)"
echo "=========================================="
