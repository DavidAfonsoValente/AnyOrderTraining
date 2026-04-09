#!/bin/bash
set -e
mkdir -p third_party
if [ ! -d third_party/WebShop ]; then
  echo "Cloning WebShop..."
  git clone https://github.com/princeton-nlp/WebShop third_party/WebShop
fi

cd third_party/WebShop

# Python 3.12 Fix: Upgrade core build tools
pip install --upgrade pip setuptools wheel --no-cache-dir

# Install dependencies
if [ -f "requirements.txt" ]; then
    echo "Installing WebShop requirements..."
    # We use --no-build-isolation to avoid pip creating a fresh environment with old setuptools
    # If this fails, we will try standard install
    pip install -r requirements.txt --no-cache-dir || pip install -r requirements.txt --no-cache-dir --no-build-isolation
fi

# Ensure it's in the python path
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Download data (small version to save quota)
if [ -f "setup.sh" ]; then
    echo "Running WebShop data setup (small)..."
    # The -d small flag is often supported in WebShop setup.sh
    bash setup.sh -d small || bash setup.sh
fi

echo "WebShop setup complete"
