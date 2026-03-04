# Any-Order Masked Training (AOMT)

This repository contains the reference implementation for the paper "Any-Order Masked Training for Trajectory-Level Supervised Learning in LLM-Based Agents". It provides all the necessary code to reproduce the experiments for the four primary baselines:

1.  **Standard SFT**: Classic next-action prediction.
2.  **Prefix SFT**: A two-stage baseline mimicking ALEE's IWM objective on offline data.
3.  **AOMT-Action-Only**: AOMT with masking restricted to action units.
4.  **AOMT-Mixed**: The full proposed method with masking over both observation and action units.

The implementation is based on the `inclusionAI/LLaDA2.0-mini` model and the `dFactory` training framework.

## 1. Repository Structure

The repository is organized as follows:

```
aomt/
├── configs/            # YAML configs for all experiments
├── data/               # Data processing scripts (download, parse, tokenize)
├── eval/               # Evaluation scripts (NLL, task-based, robustness)
├── scripts/            # Shell scripts for running training and evaluation
├── tests/              # Unit tests for core logic
├── training/           # Core training logic (sampler, objectives, trainer)
├── models/             # Directory for storing the downloaded base model
├── setup.sh            # Environment setup script
└── requirements.txt    # Python dependencies
```

## 2. Setup

The setup process involves installing dependencies, downloading the base model, and preparing the environment data.

**Prerequisites:**
*   Python 3.8+
*   `git` and `huggingface-cli`
*   An NVIDIA GPU with CUDA 12.1 or later is required for training.

**Instructions:**

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd aomt
    ```

2.  **Run the setup script (inside an interactive session):**
    The setup script downloads large model files and can consume a lot of memory. To avoid issues on shared login nodes, run it within an interactive Slurm session.

    ```bash
    # First, request a compute node
    salloc --time=1:00:00 --mem=32G --ntasks=1
    
    # Once the job starts, run the setup script
    chmod +x setup.sh
    ./setup.sh

    # After setup is complete, exit the session
    exit
    ```

3.  **Set Environment Variable (Important):**
    For the `alfworld` evaluation to work, you must set the `ALFWORLD_DATA` environment variable. The setup script will print the necessary command. Add this line to your shell configuration file (e.g., `~/.bashrc` or `~/.zshrc`) and restart your shell.

    ```bash
    # Add this line to your .bashrc or .zshrc
    export ALFWORLD_DATA=/path/to/aomt/data/alfworld
    ```

### Optional: Run Unit Tests

To ensure the core logic is functioning correctly, you can run the provided unit tests. It's recommended to do this within your activated Python virtual environment.

```bash
source venv/bin/activate
pytest aomt/tests/
```

## 3. End-to-End Workflow

This project has been automated with a set of master scripts to streamline the entire experimental process. Once the initial setup is complete, you can run all experiments and view the results by following these four steps.

### Step 1: Prepare Data

First, run the master data preparation script. This script automates the entire process of downloading, processing, and verifying the dataset. This script is idempotent and can be run safely multiple times.

```bash
srun ./scripts/prepare_data.sh
```

This script performs the following steps:
1.  **Downloads Raw Data:** Fetches the `agent-eto/eto-sft-trajectory` dataset from Hugging Face.
2.  **Processes and Tokenizes:** Converts the raw text trajectories into a memory-efficient, structured format, ready for training.
3.  **Verifies Processed Data:** Runs a verification check on the processed data, displaying features and several example `TokenizedTrajectory` objects to confirm data integrity.

### Step 2: Verify and Visualize Experiments

Before launching the full-scale experiments, you can run the master visualization script. This script provides a clear visual confirmation of the training data for each specific experiment defined in your `configs` directory.

```bash
srun ./scripts/visualize_experiments.sh
```

This script will iterate through each of your experiment config files and show you:
- The configuration being used.
- Examples of the original, unmasked training data.
- The **masked** version of the data, so you can see exactly how the data will look for that specific training run.

You can also run the full sanity check suite to verify all components:
```bash
srun ./scripts/run_verification_suite.sh
```

### Step 3: Run All Experiments on the Cluster

Next, submit all training jobs to the Slurm cluster. The `run_all_experiments.sh` script will submit the main baseline jobs in parallel and correctly handle the dependency for the two-stage Prefix-SFT experiment.

```bash
./scripts/run_all_experiments.sh
```

This will queue all the necessary training jobs. You can monitor their progress with `squeue -u $USER`. Checkpoints for each experiment will be saved to the `checkpoints/` directory.

### Step 4: Evaluate and Analyze Results

After the training jobs are complete, you need to run the full evaluation suite on each of the generated checkpoints.

First, find your checkpoint directories (e.g., `checkpoints/aomt_mixed`). Then, run the `run_full_eval.py` script for each one. This script will run all NLL, task, and robustness evaluations and save a `results.json` file inside the checkpoint directory.

**Example: Evaluating the `aomt_mixed` model**
```bash
srun python3 run_full_eval.py --checkpoint_path checkpoints/aomt_mixed
```

Once all checkpoints have been evaluated, you can automatically generate a summary table of all results using the analysis script:

```bash
python3 analysis/summarize_results.py
```

This will scan the `checkpoints` directory, find all `results.json` files, and print a comparative table of the key metrics, giving you a comprehensive overview of the experiment outcomes.
