#!/bin/bash
# Quick Start Script for Any-Order Masked Training
# This script sets up the environment, generates data, and runs a test training

set -e  # Exit on error

echo "======================================"
echo "Any-Order Masked Training Quick Start"
echo "======================================"

# Check if we're in the right directory
if [ ! -f "README.md" ]; then
    echo "Error: Please run this script from the project root directory"
    exit 1
fi

# Step 1: Install dependencies
echo ""
echo "Step 1: Installing dependencies..."
if command -v uv &> /dev/null; then
    echo "Using uv for fast installation..."
    uv venv
    source .venv/bin/activate
    uv pip install -e .
else
    echo "uv not found, using pip..."
    pip install -e .
fi

# Step 2: Install MiniGrid
echo ""
echo "Step 2: Installing MiniGrid..."
pip install minigrid gymnasium

# Step 3: Create directories
echo ""
echo "Step 3: Creating directories..."
mkdir -p data/raw data/processed
mkdir -p outputs/test_run
mkdir -p experiments

# Step 4: Generate test data
echo ""
echo "Step 4: Generating test data (200 trajectories)..."
python scripts/generate_trajectories.py \
    --env BabyAI-GoToRedBall-v0 \
    --num_episodes 200 \
    --max_steps 50 \
    --output_dir data/raw \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --seed 42

echo ""
echo "Data generation complete!"
echo "  Train: $(ls data/raw/BabyAI-GoToRedBall-v0/train/*.json | wc -l) trajectories"
echo "  Val: $(ls data/raw/BabyAI-GoToRedBall-v0/val/*.json | wc -l) trajectories"
echo "  Test: $(ls data/raw/BabyAI-GoToRedBall-v0/test/*.json | wc -l) trajectories"

# Step 5: Run test training
echo ""
echo "Step 5: Running test training (3 epochs)..."
echo "This will verify the pipeline is working correctly."
echo ""

python scripts/train.py \
    --config configs/experiments/cell_masking.yaml \
    --training.num_epochs 3 \
    --data.batch_size 4 \
    --data.data_dir data/raw \
    --output_dir outputs/test_run

echo ""
echo "======================================"
echo "Quick Start Complete!"
echo "======================================"
echo ""
echo "Next steps:"
echo "  1. Check training logs: tensorboard --logdir outputs/test_run/logs"
echo "  2. Run full experiments: see EXPERIMENTS.md"
echo "  3. Evaluate model: python scripts/evaluate.py --checkpoint outputs/test_run/checkpoints/best.pt"
echo ""
echo "For detailed instructions, see README.md and EXPERIMENTS.md"
