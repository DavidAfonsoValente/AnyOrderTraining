#!/bin/bash
set -e
mkdir -p third_party
if [ ! -d third_party/WebShop ]; then
  echo "Cloning WebShop..."
  git clone https://github.com/princeton-nlp/WebShop third_party/WebShop
fi

cd third_party/WebShop

echo "Refining WebShop for Cluster Compatibility..."

# 1. Install compatible libraries
# Downgrade pyserini slightly to avoid the 'jdk.incubator.vector' error on Java 11
pip install --no-cache-dir \
    "numpy>=1.26.0" \
    "pandas>=2.0.0" \
    "Flask>=2.3.0" \
    "gymnasium>=0.28.1" \
    "beautifulsoup4>=4.12.0" \
    "gradio>=4.0.0" \
    "gdown>=5.0.0" \
    "cleantext>=1.1.4" \
    "werkzeug>=3.0.0" \
    "rank_bm25" \
    "pyserini==0.16.0" \
    "sentence-transformers" \
    "faiss-cpu"

# 2. NEUTRALIZE their requirements.txt
if [ -f "requirements.txt" ]; then
    echo "# Cleared for compatibility" > requirements.txt
fi

# 3. Ensure path is correct
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 4. Handle Data Setup
if [ -f "setup.sh" ]; then
    echo "Attempting WebShop indexing (Lite Mode)..."
    # We ignore errors here so the rest of the project can finish
    # Most AOMT results can be gathered from ALFWorld and ScienceWorld
    bash setup.sh -d small || echo "WebShop indexing failed (likely Java/Gdown issue), continuing anyway..."
fi

echo "WebShop setup handled."
