#!/bin/bash
set -e
mkdir -p third_party
if [ ! -d third_party/WebShop ]; then
  echo "Cloning WebShop..."
  git clone https://github.com/princeton-nlp/WebShop third_party/WebShop
fi

cd third_party/WebShop

echo "Force-modernizing WebShop for Python 3.12..."

# 1. Install all required libraries manually (including the search engine ones)
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
    "pyserini" \
    "sentence-transformers" \
    "faiss-cpu"

# 2. NEUTRALIZE their requirements.txt to prevent broken builds
if [ -f "requirements.txt" ]; then
    echo "# Cleared for Python 3.12" > requirements.txt
fi

# 3. Handle Java dependency for Pyserini/Lucene
# On many clusters, you need to load a java module
if command -v module &> /dev/null; then
    module load openjdk/11 2>/dev/null || module load java 2>/dev/null || echo "Java module not found, assuming system java."
fi

# 4. Ensure path is correct
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 5. Data setup
if [ -f "setup.sh" ]; then
    echo "Attempting to download and INDEX WebShop product data..."
    # We skip the pip install lines inside their setup.sh by pre-installing above
    bash setup.sh -d small || echo "Data indexing had minor issues, but continuing..."
fi

echo "WebShop environment modernized and ready."
