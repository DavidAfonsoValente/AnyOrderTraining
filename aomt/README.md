# Any-Order Motion Transformer (AOMT)

This document provides the canonical workflow for setting up the environment, running tests, and executing the full experiment suite.

## 1. Environment Setup

This project requires **Python 3.11** and the `uv` package manager.

### Step 1: Prepare your Python 3.11 Environment

Before running the setup script, you must have a Python 3.11 executable available. There are two common ways to achieve this on a cluster:

**Option A: Using Conda (Recommended)**
If you have Conda or Miniconda installed, create and activate a Python 3.11 environment:
```bash
conda create -n py311 python=3.11 -y
conda activate py311
```

**Option B: Using Environment Modules**
If your cluster uses environment modules, load Python 3.11:
```bash
# The exact module name may vary
module load python/3.11
```

### Step 2: Run the Project Setup Script

Once your shell is configured with Python 3.11, run the main setup script from the `aomt/` directory. This script will install `uv` (if needed), clone all dependencies, and set up the project's virtual environment.

```bash
# This only needs to be run once.
./setup.sh
```
This script will create a helper script, `activate_env.sh`, for convenient activation.

### Step 3: Activate the Project Environment

For all subsequent steps (running tests, data prep, training), you must be in the correct environment. From the `aomt/` directory, run:
```bash
source activate_env.sh
```
This will activate both the Python module (if needed) and the project-specific virtual environment.

## 2. Data Preparation (on a Compute Node)

After activating the environment, process the raw dataset. This is memory-intensive and **must be run on a compute node**.

1.  **Request an interactive Slurm session:**
    ```bash
    salloc --time=1:00:00 --mem=128G --gpus=1 --ntasks=1
    ```

2.  **Activate the environment (inside the Slurm session):**
    ```bash
    # Navigate to the aomt/ directory first
    source activate_env.sh
    ```

3.  **Run the data preparation script:**
    ```bash
    ./scripts/prepare_data.sh
    ```

## 3. Running the Test Suite

From the `aomt/` directory, with the environment activated (`source activate_env.sh`), run the test suite:

```bash
./scripts/run_tests.sh
```

## 4. Running Experiments

From the `aomt/` directory on a login node, with the environment activated (`source activate_env.sh`), execute the master script to submit all training jobs:

```bash
./scripts/run_all_experiments.sh
```
You can monitor the jobs with `squeue -u $USER`.
