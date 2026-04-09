#!/bin/bash
set -e

# Detect if we are on a compute node and set up local storage if so
if [[ $(hostname) != xlogin* ]]; then
    echo "Compute node detected. Using local SSD (/tmp) to bypass home quota."
    export TMPDIR=/tmp/$USER
    export HF_HOME=/tmp/$USER/hf_cache
    export PIP_CACHE_DIR=/tmp/$USER/pip_cache
    mkdir -p $TMPDIR $HF_HOME $PIP_CACHE_DIR
fi

echo "Starting AOMT environment setup..."

# 1. Install Python requirements
pip install --no-cache-dir -r requirements.txt

# 2. Setup dFactory
if [ ! -d "aomt/dFactory" ]; then
    git clone https://github.com/inclusionAI/dFactory.git aomt/dFactory
fi
cd aomt/dFactory
git submodule update --init --recursive
cd ../..

# 3. Setup benchmarks
bash scripts/setup_alfworld.sh
bash scripts/setup_scienceworld.sh
bash scripts/setup_webshop.sh

# 4. Download ETO Trajectory Data
python scripts/download_data.py

# 5. Download Model Weights (32GB)
echo "Downloading LLaDA 2.0-mini weights..."
python scripts/download_model.py

echo "Full setup complete. You can now exit and run experiments."
