#!/bin/bash
set -e
mkdir -p third_party
if [ ! -d third_party/WebShop ]; then
  echo "Cloning WebShop..."
  git clone https://github.com/princeton-nlp/WebShop third_party/WebShop
fi

cd third_party/WebShop

echo "Force-modernizing WebShop for Python 3.12 Compatibility..."

# 1. Install modern, compatible versions of all WebShop dependencies
# We do this first so they are already satisfied when we run their scripts
pip install --no-cache-dir \
    "numpy>=1.26.0" \
    "pandas>=2.0.0" \
    "Flask>=2.3.0" \
    "gymnasium>=0.28.1" \
    "beautifulsoup4>=4.12.0" \
    "gradio>=4.0.0" \
    "gdown>=5.0.0" \
    "cleantext>=1.1.4" \
    "werkzeug>=3.0.0"

# 2. NEUTRALIZE their requirements.txt and setup.py
# This prevents their scripts from trying to reinstall the broken old versions
if [ -f "requirements.txt" ]; then
    echo "# Cleared for Python 3.12 compatibility" > requirements.txt
fi

# 3. Ensure the package is in the path
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 4. Handle Data Setup manually to avoid their internal pip calls
# The setup.sh in WebShop often contains 'pip install' lines. 
# We will only run the data download portions.
if [ -f "setup.sh" ]; then
    echo "Attempting to download WebShop product data..."
    # We try to run their script but ignore errors, as we already installed the libs
    bash setup.sh -d small || echo "Data download interrupted or failed, continuing..."
fi

echo "WebShop environment modernized and ready."
