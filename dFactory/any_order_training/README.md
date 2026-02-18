
# Any-Order Masked Training for Trajectory-Level Learning

This project implements the "Any-Order Masked Training" paradigm for LLM-based agents. It is built on top of the `dFactory` and `VeOmni` framework to fine-tune the LLaDA 2.0 model.

## Quick Start: Automated Setup, Download, and Test

A single script is provided to automate the entire setup, model download, data generation, and testing process.

**Usage:**

From the root of the `dFactory` directory, run the script with a single argument: the path to your desired output directory.

```bash
bash any_order_training/setup_and_test.sh /path/to/your/output_directory
```

**Example:**
```bash
bash any_order_training/setup_and_test.sh /home/d/dvalente/AnyOrderTraining/output
```

**What this script does:**

1.  **Environment Setup**: Creates a Python virtual environment and installs all necessary dependencies (`uv`, `gymnasium`, `minigrid`, `huggingface_hub`, etc.).
2.  **Model Download**: Submits a SLURM job (`any_order_training/slurm/download_model.sbatch`) to download the `inclusionAI/LLaDA2.0-mini-preview` model and convert it to the required "merged-expert" format. The script will wait for this job to complete before continuing. The model is saved to `/scratch/$USER/models` if available, otherwise `~/models`.
3.  **Data Generation**: Generates the BabyAI trajectory datasets.
4.  **Configuration**: Automatically configures the smoke test to use the downloaded model path and your specified output path.
5.  **Local Tests**: Runs the local unit tests.
6.  **GPU Smoke Test**: Submits a smoke test job to your SLURM cluster to verify that the end-to-end training pipeline works on a GPU.

After running this script, your environment will be fully set up, the model downloaded, data generated, and a test run will be initiated.

## Manual Usage

For more granular control over the process (e.g., to run different experiments), you can follow the manual steps. Please refer to the `PROJECT_DESCRIPTION.md` and `testing.md` files for more details.

## Project Structure

The project is organized as follows:

```
any_order_training/
├── setup_and_test.sh           # Automated setup and testing script
├── configs/
│   └── ...                     # Configuration files for experiments
├── data/
│   └── ...                     # Data generation and processing
├── sampler/
│   └── ...                     # The AnyOrderMaskSampler
├── scripts/
│   └── ...                     # Evaluation script
├── slurm/
│   └── ...                     # SLURM batch scripts
├── tasks/
│   └── ...                     # The main training script
└── tests/
    └── ...                     # Unit tests
```
