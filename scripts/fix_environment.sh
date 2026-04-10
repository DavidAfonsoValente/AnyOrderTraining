#!/bin/bash
# scripts/fix_environment.sh
set -e

PROJECT_DIR="/home/d/dvalente/AnyOrderTraining"
cd "$PROJECT_DIR"

source venv/bin/activate

echo "Ensuring Pip is ready..."
pip install --upgrade pip

echo "Installing Cluster-Compatible PyTorch (CUDA 12.1)..."
# We use /tmp for the cache to bypass your home directory quota
export PIP_CACHE_DIR="/tmp/$USER/pip_cache"
mkdir -p "$PIP_CACHE_DIR"

pip install --no-cache-dir --force-reinstall \
    torch==2.4.0 \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

echo "------------------------------------------------"
echo "VERIFYING CUDA VISIBILITY:"
python3 -c "import torch; print(f'RESULT: CUDA Available = {torch.cuda.is_available()}'); print(f'DEVICE: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'RESULT: FAILED')"
echo "------------------------------------------------"
