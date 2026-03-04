#!/bin/bash
# AOMT Environment Setup Script

# Exit immediately if a command exits with a non-zero status.
set -e

# --- 1. Install Python Dependencies ---
echo "Installing Python packages from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

# --- 2. Install dFactory ---
echo "Cloning and installing dFactory..."
if [ -d "dFactory" ]; then
    echo "dFactory directory already exists. Skipping clone."
else
    git clone https://github.com/inclusionAI/dFactory
fi
cd dFactory
pip install -e .
cd ..
echo "dFactory installation complete."

# --- 3. Download LLaDA 2.0 mini model ---
echo "Downloading LLaDA 2.0 mini model..."
if [ -d "models/LLaDA2.0-mini" ]; then
    echo "Model directory already exists. Skipping download."
else
    # The spec uses a generic path, we'll place it inside the aomt project folder
    huggingface-cli download inclusionAI/LLaDA2.0-mini --local-dir ./models/LLaDA2.0-mini --local-dir-use-symlinks False
fi
echo "LLaDA 2.0 mini model downloaded."


# --- 4. Setup Evaluation Data ---
echo "Setting up evaluation environment data..."
# Set ALFWORLD_DATA environment variable. Note: This needs to be set in the user's shell profile for persistence.
export ALFWORLD_DATA=$(pwd)/data/alfworld
echo "ALFWORLD_DATA will be set to: $ALFWORLD_DATA"
echo "To make this permanent, please add 'export ALFWORLD_DATA=$(pwd)/data/alfworld' to your ~/.bashrc or ~/.zshrc file."
# The alfworld library will automatically download its data to this location upon first use.
# We create the directory to be safe.
mkdir -p $ALFWORLD_DATA
echo "Evaluation data setup initiated."

echo -e "
--- Environment setup complete! ---"
echo "Remember to set the ALFWORLD_DATA environment variable in your shell profile."

