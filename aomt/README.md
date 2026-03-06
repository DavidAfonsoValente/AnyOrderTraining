# Any-Order Motion Transformer (AOMT)

This repository contains the implementation for training and evaluating the Any-Order Motion Transformer, a model for learning robotic manipulation policies.

## 1. Setup

These steps should be performed on the cluster's login node (e.g., `xlogin1`).

### Step 1.1: Clone Repository

Clone this repository and its submodules (`dFactory`/`VeOmni`).

```bash
git clone --recurse-submodules <repository_url>
cd aomt
```

### Step 1.2: Create Virtual Environment

Create and activate a Python virtual environment.

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 1.3: Install Dependencies

Install the required Python packages.

```bash
pip install -r requirements.txt
```

### Step 1.4: Download Pre-trained Model

The training framework requires the pre-trained `LLaDA2.0-mini` model to be present locally. Run the provided download script. This will download the model into `aomt/models/LLaDA2.0-mini/`.

```bash
python3 dFactory/scripts/download_hf_model.py --repo_id inclusionAI/LLaDA2.0-mini --local_dir ./models
```

### Step 1.5: Prepare the Dataset

Parse the raw trajectory data into the processed format used for training. This script only needs to be run once.

```bash
python3 data/parse_trajectories.py
```

## 2. Running Tests

Before launching a full-scale experiment, run the complete test suite to verify that all components are working correctly. This includes unit tests and an end-to-end integration test.

Execute this command from the `aomt/` directory:

```bash
python3 -m unittest discover tests
```

## 3. Running Experiments

Submit all training jobs to the Slurm cluster. The master script will queue all baseline experiments and correctly handle dependencies for the multi-stage jobs.

```bash
./scripts/run_all_experiments.sh
```

You can monitor the status of your jobs using:
```bash
squeue -u $USER
```

## 4. Evaluation

After your training jobs are complete, you can run the evaluation suite on the generated checkpoints located in the `checkpoints/` directory using the scripts in `aomt/scripts/` (e.g., `run_full_eval.py`).
