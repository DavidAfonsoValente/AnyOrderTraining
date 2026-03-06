#!/bin/bash
# Full Setup Script: Installs Miniconda, creates Python 3.11 env, then runs project setup.
set -e

echo "--- Full Project Setup (including Miniconda installation) ---"

# --- 1. Install Miniconda (if not already installed) ---
if [ ! -d "$HOME/miniconda3" ]; then
    echo "[1/4] Installing Miniconda..."
    mkdir -p "$HOME/miniconda3"
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$HOME/miniconda3/miniconda.sh"
    bash "$HOME/miniconda3/miniconda.sh" -b -u -p "$HOME/miniconda3"
    rm -rf "$HOME/miniconda3/miniconda.sh"
    # Initialize conda for the current shell session
    source "$HOME/miniconda3/bin/activate" || true
    conda init bash || true
    echo "Miniconda installed."
else
    echo "[1/4] Miniconda already installed."
    source "$HOME/miniconda3/bin/activate" || true
fi

# Ensure conda commands are available in the current shell context
if [ -f "$HOME/.bashrc" ]; then source "$HOME/.bashrc" || true; fi

# --- 2. Accept Conda Terms of Service ---
echo "[2/4] Accepting Conda Terms of Service..."
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true
echo "Terms of Service accepted."

# --- 3. Create and Activate Python 3.11 Conda Environment ---
echo "[3/4] Creating and activating Python 3.11 Conda environment..."
CONDA_ENV_NAME="py311"
if ! conda env list | grep -q "$CONDA_ENV_NAME"; then
    echo "Creating conda environment '$CONDA_ENV_NAME' with Python 3.11..."
    conda create -n "$CONDA_ENV_NAME" python=3.11 -y
else
    echo "Conda environment '$CONDA_ENV_NAME' already exists."
fi
conda activate "$CONDA_ENV_NAME"
echo "Conda environment '$CONDA_ENV_NAME' activated."

# --- 4. Run Project-Specific Setup ---
echo "[4/4] Running project-specific setup (./setup.sh)..."
# The main setup.sh script is assumed to be in the same directory
chmod +x ./setup.sh
./setup.sh

echo -e "
--- Full setup complete! ---"
echo "The script has finished. To use the environment in a new terminal:"
echo "1. Activate the Conda environment: conda activate $CONDA_ENV_NAME"
echo "2. Navigate to this directory ('~/AnyOrderTraining/aomt')."
echo "3. Activate the project-specific environment: source activate_env.sh"
