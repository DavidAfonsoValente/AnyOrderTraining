#!/bin/bash
# AOMT Environment Setup Script

# Exit immediately if a command exits with a non-zero status.
set -e

# --- 0. Setup Virtual Environment ---
VENV_DIR="venv"
echo "--- Setting up Python virtual environment in './${VENV_DIR}' ---"
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists. Skipping creation."
else
    python3 -m venv "$VENV_DIR"
fi

# Activate the virtual environment
source "${VENV_DIR}/bin/activate"
echo "Virtual environment activated."


# --- 1. Install Python Dependencies ---
echo "Installing Python packages from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

# --- 2. Clone dFactory repository ---
echo "Cloning dFactory repository..."
if [ -d "dFactory" ]; then
    echo "dFactory directory already exists. Skipping clone."
else
    git clone https://github.com/inclusionAI/dFactory
fi
echo "dFactory repository is ready."

# --- 3. Download LLaDA 2.0 mini model ---
echo "Downloading LLaDA 2.0 mini model..."
# A more robust check for a key file, not just the directory.
if [ -f "models/LLaDA2.0-mini/model.safetensors.index.json" ]; then
    echo "Model appears to be downloaded already (found model.safetensors.index.json). Skipping download."
else
    # The spec uses a generic path, we'll place it inside the aomt project folder
    huggingface-cli download inclusionAI/LLaDA2.0-mini --local-dir ./models/LLaDA2.0-mini --local-dir-use-symlinks False
fi
echo "LLaDA 2.0 mini model download check complete."


# --- 4. Setup Evaluation Data ---
echo "Setting up evaluation environment data..."
# Set ALFWORLD_DATA environment variable. Note: This needs to be set in the user's shell profile for persistence.
export ALFWORLD_DATA=$(pwd)/data/alfworld
echo "ALFWORLD_DATA will be set to: $ALFWORLD_DATA"
echo "To make this permanent, please add the following to your ~/.bashrc or ~/.zshrc file:"
echo "export ALFWORLD_DATA=\\"$(pwd)/data/alfworld\\""
# The alfworld library will automatically download its data to this location upon first use.
# We create the directory to be safe.
mkdir -p $ALFWORLD_DATA
echo "Evaluation data setup initiated."

echo -e "\n--- Environment setup complete! ---"
echo "To use this environment in your shell, run: source ${VENV_DIR}/bin/activate"
