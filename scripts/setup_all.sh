#!/bin/bash
set -e

echo "Starting AOMT environment setup..."

# 1. Install Python requirements
pip install -r requirements.txt

# 2. Clone dFactory if not present
if [ ! -d "aomt/dFactory" ]; then
    echo "Cloning dFactory..."
    git clone https://github.com/inclusionAI/dFactory.git aomt/dFactory
fi

# 3. Initialize dFactory submodules
echo "Initializing dFactory submodules..."
cd aomt/dFactory
git submodule update --init --recursive
cd ../..

# 4. Setup benchmarks
echo "Setting up benchmarks..."
bash scripts/setup_alfworld.sh
bash scripts/setup_scienceworld.sh
bash scripts/setup_webshop.sh

# 5. Download data
echo "Downloading ETO dataset..."
python scripts/download_data.py

echo "Full setup complete."
