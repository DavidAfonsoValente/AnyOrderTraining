#!/bin/bash
set -e
mkdir -p third_party
if [ ! -d third_party/WebShop ]; then
  echo "Cloning WebShop..."
  git clone https://github.com/princeton-nlp/WebShop third_party/WebShop
fi

cd third_party/WebShop

# Check if setup.py exists, if not, install dependencies manually
if [ -f "setup.py" ]; then
    pip install -e . --no-cache-dir
else
    echo "setup.py not found in WebShop root. Installing requirements manually..."
    # Install standard requirements
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt --no-cache-dir
    fi
    # Ensure it's in the python path
    export PYTHONPATH=$PYTHONPATH:$(pwd)
fi

# Download data (this can be large, watch your quota)
# Some versions of WebShop use a setup.sh for data
if [ -f "setup.sh" ]; then
    bash setup.sh
fi

echo "WebShop setup complete"
