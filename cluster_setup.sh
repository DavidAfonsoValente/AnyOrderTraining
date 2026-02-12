#!/bin/bash
# Setup script for SoC Slurm Cluster
# Run this on the login node (xlogin0/1/2)

set -e

echo "=========================================="
echo "Any-Order Training - Cluster Setup"
echo "=========================================="

# Create virtual environment
echo "Creating Python virtual environment..."
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -e .
pip install minigrid gymnasium
pip install numpy pandas pyyaml tqdm wandb tensorboard
pip install matplotlib seaborn scikit-learn
pip install transformers accelerate

# Create necessary directories
echo "Creating directories..."
mkdir -p data/raw data/processed
mkdir -p outputs logs
mkdir -p experiments

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Virtual environment created at: ./venv"
echo ""
echo "To activate the environment manually:"
echo "  source venv/bin/activate"
echo ""
echo "Next steps:"
echo "  1. Test data generation: sbatch scripts/slurm/test_datagen.sh"
echo "  2. Quick pipeline test: sbatch scripts/slurm/test_pipeline.sh"
echo "  3. Full experiments: bash scripts/slurm/submit_all.sh"
