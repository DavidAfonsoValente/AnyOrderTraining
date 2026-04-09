#!/bin/bash
set -e
mkdir -p third_party
if [ ! -d third_party/WebShop ]; then
  echo "Cloning WebShop..."
  git clone https://github.com/princeton-nlp/WebShop third_party/WebShop
fi

cd third_party/WebShop

echo "Python 3.12 detected. Modernizing WebShop dependencies..."

# 1. Upgrade build tools again
pip install --upgrade pip setuptools wheel --no-cache-dir

# 2. Install modern versions of the broken requirements
# We ignore the versions in requirements.txt and use versions that build on 3.12
pip install "numpy>=1.26.0" "Flask>=2.3.0" "gymnasium>=0.28.1" --no-cache-dir

# 3. Install the remaining requirements, filtering out the ones we just handled
if [ -f "requirements.txt" ]; then
    grep -vE "numpy|Flask|gym|setuptools" requirements.txt > requirements_312.txt
    echo "Installing filtered WebShop requirements..."
    pip install -r requirements_312.txt --no-cache-dir || echo "Some minor WebShop dependencies failed, continuing..."
fi

# 4. Ensure path is correct
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 5. Data setup (minimal)
if [ -f "setup.sh" ]; then
    echo "Running WebShop data setup (small)..."
    # Try to run setup but don't fail the whole script if download fails
    bash setup.sh -d small || echo "WebShop data download failed/interrupted, skipping for now."
fi

echo "WebShop setup complete (Modernized for Python 3.12)"
