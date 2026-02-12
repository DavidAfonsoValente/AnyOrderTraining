#!/bin/bash
#SBATCH --job-name=test_pipeline
#SBATCH --time=30
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/test_pipeline_%j.out
#SBATCH --error=logs/test_pipeline_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dvalente@comp.nus.edu.sg

# Activate virtual environment
source venv/bin/activate

echo "=========================================="
echo "Quick Pipeline Test"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "=========================================="

# Check if test data exists, if not generate it
if [ ! -d "data/test/BabyAI-GoToRedBall-v0" ]; then
    echo "Test data not found, generating..."
    python scripts/generate_trajectories.py \
        --env BabyAI-GoToRedBall-v0 \
        --num_episodes 100 \
        --max_steps 30 \
        --output_dir data/test \
        --train_ratio 0.7 \
        --val_ratio 0.15 \
        --seed 42
fi

# Run quick training test (1 epoch, CPU)
echo ""
echo "Running pipeline test (1 epoch)..."
python scripts/train.py \
    --config configs/experiments/cell_masking.yaml \
    --training.num_epochs 1 \
    --data.batch_size 2 \
    --data.data_dir data/test \
    --output_dir outputs/test_pipeline_${SLURM_JOB_ID}

echo ""
echo "=========================================="
echo "Pipeline test complete!"
echo "Outputs in: outputs/test_pipeline_${SLURM_JOB_ID}/"
echo "Finished: $(date)"
echo "=========================================="
