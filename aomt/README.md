# Any-Order Masked Training (AOMT)

This repository contains the implementation for "Any-Order Masked Training for Trajectory-Level Supervised Learning in LLM-Based Agents," a supervised fine-tuning (SFT) paradigm that reframes agent learning as a trajectory-level reconstruction problem. It uses a masked diffusion language model to reconstruct arbitrarily masked observation/action units from bidirectional context.

This document provides the canonical workflow for setting up the environment, preparing data, and running experiments.

---

## 1. Setup and Workflow

The setup is a **3-step process**. You will first set up the environment on a login node, then prepare the model, and finally prepare the dataset on a compute node.

**Prerequisite:** You must be logged into the Hugging Face Hub. If you are not, run the following command and provide your access token:
```bash
huggingface-cli login
```

### Step 1: Environment Setup (Login Node)

From the `aomt/` directory, run the main setup script. This script will prepare your environment by installing all necessary dependencies, cloning the `dFactory` framework, and downloading the `LLaDA-2.0-mini` base model.

```bash
./setup.sh
```
This script creates the virtual environment and a helper script, `activate_env.sh`, for easy activation.

### Step 2: Model Preparation (Login Node)

The `dFactory` framework requires the model's weights to be in a special 'merged-expert' format for efficient training. After setting up the environment, you must run the model preparation script.

First, activate the environment you just created:
```bash
source activate_env.sh
```

Then, run the preparation script:
```bash
./scripts/prepare_model.sh
```
This will create a new directory, `aomt/models/LLaDA2.0-mini-merged`, containing the model weights in the correct format for training.

**Important:** Before running training, you must update your experiment config files in `aomt/configs/` to point to this new merged model path. For example, change `model_path: models/LLaDA2.0-mini` to `model_path: models/LLaDA2.0-mini-merged`.

### Step 3: Data Preparation (Compute Node)

This step downloads and processes the `agent-eto/eto-sft-trajectory` dataset. It is memory-intensive and **must be run on a compute node.**

1.  **Request an interactive Slurm session:**
    ```bash
    salloc --time=2:00:00 --mem=128G --gpus=1 --ntasks=1
    ```

2.  **Navigate to the `aomt/` directory and activate the environment:**
    ```bash
    cd /path/to/your/AnyOrderTraining/aomt
    source activate_env.sh
    ```

3.  **Run the data preparation script:**
    You must use `srun` to ensure the script executes on the allocated compute node.
    ```bash
    srun ./scripts/prepare_data.sh
    ```
    This will download the raw dataset to `aomt/data/dataset_cache/raw` and then process it into `aomt/data/dataset_cache/train` and `aomt/data/dataset_cache/validation`.

---

## 2. Running the Project

For all subsequent work (running tests, training, evaluation), you must first activate the environment.

### Activating the Environment

From the `aomt/` directory, run:
```bash
source activate_env.sh
```
This will activate the correct virtual environment and set all necessary paths.

### Running the Test Suite

With the environment activated, run the test suite to verify the core components:
```bash
./scripts/run_tests.sh
```

### Running Experiments

From a login node, with the environment activated, execute the master script to submit all training jobs to the Slurm scheduler:
```bash
./scripts/run_all_experiments.sh
```
You can monitor your jobs with `squeue -u $USER`. The script will submit separate jobs for each experiment configuration defined in the `aomt/configs/` directory. Results and checkpoints will be saved to the path configured in the respective YAML files.
