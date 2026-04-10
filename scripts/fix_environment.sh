#!/bin/bash
# scripts/fix_environment.sh
set -e

PROJECT_DIR="/home/d/dvalente/AnyOrderTraining"
cd "$PROJECT_DIR"

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

echo "Updating pip..."
pip install --upgrade pip

echo "Force-installing Cluster-Compatible PyTorch (CUDA 12.1)..."
# This version matches the '12090' driver found on your xgpi nodes
pip install --no-cache-dir --force-reinstall torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "Verifying CUDA visibility..."
python3 -c "import torch; print(f'RESULT: CUDA Available = {torch.cuda.is_available()}'); print(f'DEVICE: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'RESULT: FAILED')"

echo "------------------------------------------------"
echo "If RESULT is True, your environment is FIXED."
echo "You can now exit this node and run the pipeline."
