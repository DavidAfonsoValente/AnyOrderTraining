# Any-Order Motion Transformer (AOMT)

This document provides the canonical workflow for setting up the environment, running tests, and executing the full experiment suite.

## 1. Project Overview

As described in `project_desc.md`, this project implements **Any-Order Masked Training (AOMT)**. It reframes agent fine-tuning as a trajectory reconstruction problem, using a masked diffusion model (`LLaDA 2.0-mini`) to reconstruct missing trajectory segments from bidirectional context. The goal is to improve generalization and world-model understanding using a fixed expert dataset (`agent-eto/eto-sft-trajectory`).

## 2. Setup and Data Preparation

The setup is a two-phase process. Phase 1 can be run on a login node, while Phase 2 must be run on a compute node due to high memory requirements.

### Phase 1: Initial Setup (on Login Node)

This is handled by the `setup.sh` script, which installs dependencies and downloads the required pre-trained model.

```bash
# Run from the 'aomt/' directory
chmod +x setup.sh
./setup.sh
```

### Phase 2: Data Processing (on Compute Node)

This step processes the raw dataset into the format required for training. **It only processes the `train` split**, as the `test` split in the source data is corrupt and causes fatal errors. All evaluation is performed on a validation set held out from the `train` data.

1.  **Request an interactive compute node:**
    ```bash
    salloc --time=1:00:00 --mem=128G --gpus=1 --ntasks=1
    ```

2.  **Run the data preparation script:**
    Once the job starts, run the script. It will now succeed by only processing the `train` split.
    ```bash
    ./scripts/prepare_data.sh
    ```

3.  **Set Environment Variable:**
    Finally, add the following line to your `~/.bashrc` or `~/.zshrc` on the login node to configure the evaluation environment.
    ```bash
    export ALFWORLD_DATA="$(pwd)/data/alfworld"
    ```

## 3. End-to-End Workflow

### Step 3.1: Run Tests

Before launching experiments, verify the setup by running the test suite. This script handles all pathing and environment setup automatically.

```bash
# Run from the 'aomt/' directory
./scripts/run_tests.sh
```

### Step 3.2: Run Experiments

The `run_all_experiments.sh` script submits all training jobs to the Slurm scheduler. You can edit the `CONFIG_CHAIN` array inside this script to control which experiments are run.

```bash
# Run from the 'aomt/' directory
./scripts/run_all_experiments.sh
```
Monitor jobs with `squeue -u $USER`.

### Step 3.3: Evaluate and Analyze

After training, evaluate your models. This should be done on a compute node.

```bash
# Get a compute node
salloc --time=0:30:00 --mem=32G --gpus=1 --ntasks=1

# Run evaluation on a generated checkpoint
# The script now uses a validation set from the 'train' data
python3 run_full_eval.py --checkpoint_path checkpoints/aomt_mixed

# After evaluating, summarize the results on the login node
python3 analysis/summarize_results.py
```
