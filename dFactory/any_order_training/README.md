# Any-Order Masked Training for Trajectory-Level Learning

This project implements the "Any-Order Masked Training" paradigm for LLM-based agents. It is built on top of the `dFactory` and `VeOmni` framework to fine-tune the LLaDA 2.0 model.

## 1. Getting the LLaDA 2.0 Model

This project requires the LLaDA 2.0 model for fine-tuning. The training scripts are optimized for a "merged-expert" format of the model weights.

### Which Model to Use

For this project, we recommend starting with the smaller **`inclusionAI/LLaDA2.0-mini-preview`** (16B) model. It is more manageable for fine-tuning and experimentation.

### How to Get and Prepare the Model

Follow these steps to download the model from the Hugging Face Hub and convert it to the required format.

**Step 1: Download the original model**

The `dFactory` framework provides a helper script to download the model. Choose a directory where you want to save the model, for example `/home/user/llada_models/original`.

```bash
python scripts/download_hf_model.py \
  --repo_id inclusionAI/LLaDA2.0-mini-preview \
  --local_dir /path/to/save/original_model # e.g., /home/user/llada_models/original
```

**Step 2: Convert to the "merged-expert" format**

The training scripts require the model's expert weights to be merged into a single tensor for efficiency. Use the following script to perform this conversion.

The output of this step will be the model you use for training.

```bash
# Use the path from the previous step as the input
python scripts/moe_convertor.py \
  --input-path /path/to/save/original_model \
  --output-path /path/to/save/merged_model \ # e.g., /home/user/llada_models/merged
  --mode merge
```

The directory `/path/to/save/merged_model` is the one you will use as the `model_path` in the next steps.

## 2. Automated Setup and Testing

Once you have the model, you can use the provided script to automate the rest of the setup and testing process.

**Usage:**

From the root of the `dFactory` directory, run:

```bash
bash any_order_training/setup_and_test.sh /path/to/your/merged_model /path/to/your/output_directory
```

**Arguments:**

1.  `/path/to/your/merged_model`: The absolute path to the directory containing the **merged** LLaDA 2.0 model (from step 2 above).
2.  `/path/to/your/output_directory`: The absolute path to a directory where training outputs will be saved (e.g., `output/`).

**What this script does:**

1.  **Environment Setup**: Creates a virtual environment and installs all necessary dependencies.
2.  **Data Generation**: Generates the BabyAI trajectory datasets.
3.  **Configuration**: Automatically configures the smoke test to use the paths you provide.
4.  **Local Tests**: Runs the local unit tests.
5.  **GPU Smoke Test**: Submits a smoke test job to your SLURM cluster to verify that the end-to-end training pipeline works on a GPU.

## 3. Manual Usage

For more control over the process, you can follow the manual steps for setup, data generation, configuration, and training. Please refer to the `PROJECT_DESCRIPTION.md` and `testing.md` files for more details on the project and the manual testing procedures.

## 4. Project Structure

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
