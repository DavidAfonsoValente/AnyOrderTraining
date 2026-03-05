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
    # First, request a compute node if you don't have an active one
    salloc --time=1:00:00 --mem=32G --ntasks=1 --gres=gpu:1 # Request a GPU for setup if needed
    
    # Once the job starts (e.g., you're on a compute node), run the setup script
    chmod +x setup.sh
    ./setup.sh

    # After setup is complete, you can exit the allocation
    exit
    ```

3.  **Set Environment Variable (Important):**
    For the `alfworld` evaluation to work, you must set the `ALFWORLD_DATA` environment variable. The setup script will print the necessary command. Add this line to your shell configuration file (e.g., `~/.bashrc` or `~/.zshrc`) and restart your shell.

    ```bash
    # Add this line to your .bashrc or .zshrc
    export ALFWORLD_DATA=/path/to/aomt/data/alfworld
    ```

### Optional: Run Unit Tests

To ensure the core logic is functioning correctly, you can run the provided unit tests. It's recommended to do this within an active Slurm allocation, as some tests might involve loading models or heavy computation.

```bash
# First, request a compute node if you don't have an active one
salloc --time=0:15:00 --mem=16G --ntasks=1 # Adjust resources as needed

# Once on the compute node, activate your venv and run tests
source venv/bin/activate
srun python -m pytest aomt/tests/
```

### Optional: Weights & Biases (WandB) Setup

For online monitoring of training runs, you can use Weights & Biases.
1.  Ensure `wandb` is installed (it's in `requirements.txt`).
2.  Log in to your WandB account in your terminal (do this from a login node or compute node):
    ```bash
    wandb login
    ```
Training runs will automatically log to your WandB dashboard.

## 3. End-to-End Workflow

This project has been automated with a set of master scripts to streamline the entire experimental process. Once the initial setup is complete, you can run all experiments and view the results by following these four steps.

\textbf{Note}: All commands involving `srun` should be executed from within an active SLURM allocation (e.g., requested via `salloc` or submitted as part of a batch job script).

### Step 1: Prepare Data

First, run the master data preparation script inside an interactive Slurm session. This script automates the entire process of downloading, processing, and verifying the dataset. This script is idempotent and can be run safely multiple times.

```bash
# First, request a compute node if you don't have an active one
salloc --time=1:00:00 --mem=32G --ntasks=1 --gres=gpu:1 # Data prep can be resource intensive

# Once on the compute node, activate your venv and run the script
source venv/bin/activate
srun ./scripts/prepare_data.sh
```

### Step 2: Verify and Visualize Experiments

Before launching the full-scale experiments, it's highly recommended to run the verification and visualization scripts.

**A) Run Automated Sanity Checks:**
To perform a suite of automated checks on the core components of the model and data processing, run the `run_verification_suite.sh` script. This script now runs to completion, providing warnings for non-critical issues.

```bash
# First, request a compute node
salloc --time=0:15:00 --mem=16G --ntasks=1 --gres=gpu:1 # Adjust resources as needed

# Once on the compute node, activate your venv and run the script
source venv/bin/activate
srun ./scripts/run_verification_suite.sh
```
This script checks attention masks, loss functions, and other critical components to catch potential issues early.

**B) Visualize Data for Each Experiment:**
To see concrete examples of how the data is prepared for each training run, use the `visualize_experiments.sh` script. This script has been significantly improved to show clear, untruncated examples for all modes.

```bash
# First, request a compute node
salloc --time=0:15:00 --mem=16G --ntasks=1 --gres=gpu:1 # Adjust resources as needed

# Once on the compute node, activate your venv and run the script
source venv/bin/activate
srun ./scripts/visualize_experiments.sh
```

This will loop through each of your experiment configs (e.g., `sft_standard.yaml`, `aomt_mixed.yaml`) and show you a few examples of the exact masked data the model will see during training, making it easy to confirm each experiment is set up as intended.

### Step 3: Run All Experiments on the Cluster

Next, submit all training jobs to the Slurm cluster. The `run_all_experiments.sh` script will submit the main baseline jobs in parallel and correctly handle the dependency for the two-stage Prefix-SFT experiment. Training runs will automatically log to WandB for online monitoring.

```bash
# This script uses 'sbatch' internally to submit jobs.
# You typically run this from a login node (no salloc needed around it).
# Activate your venv if needed for submitting script:
source venv/bin/activate
./scripts/run_all_experiments.sh
```

This will queue all the necessary training jobs. You can monitor their progress with `squeue -u $USER` and online via your WandB dashboard. Checkpoints for each experiment will be saved to the `checkpoints/` directory.

### Step 4: Evaluate and Analyze Results

After the training jobs are complete, you need to run the full evaluation suite on each of the generated checkpoints.

**A) Run Zero-shot Baseline Evaluation:**
To establish a performance baseline for the untuned model:
```bash
# First, request a compute node
salloc --time=0:30:00 --mem=32G --ntasks=1 --gres=gpu:1 # Adjust resources as needed

# Once on the compute node, activate your venv and run the script
source venv/bin/activate
srun ./scripts/run_zeroshot_eval.sh
```

**B) Evaluate Trained Models:**
First, find your checkpoint directories (e.g., `checkpoints/aomt_mixed`). Then, run the `run_full_eval.py` script for each one. This script will run all NLL, task, and robustness evaluations and save a `results.json` file inside the checkpoint directory.

```bash
# First, request a compute node for each evaluation job
salloc --time=0:30:00 --mem=32G --ntasks=1 --gres=gpu:1 # Adjust resources as needed

# Once on the compute node, activate your venv and run the evaluation
source venv/bin/activate
python3 run_full_eval.py --checkpoint_path checkpoints/aomt_mixed
```

Once all checkpoints have been evaluated, you can automatically generate a summary table of all results using the analysis script:

```bash
# This script typically runs on a login node as it's not compute intensive
source venv/bin/activate
python3 analysis/summarize_results.py
```

This will scan the `checkpoints` directory, find all `results.json` files, and print a comparative table of the key metrics, giving you a comprehensive overview of the experiment outcomes.
