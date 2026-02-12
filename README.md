# Any-Order Masked Training for Trajectory-Level Learning

This project implements **Any-Order Masked Training**, a supervised fine-tuning paradigm for LLM-based agents that treats agent learning as a trajectory-level reconstruction problem.

## Project Structure

```
any-order-training/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── setup.py                          # Package installation
├── configs/                          # Configuration files
│   ├── base_config.yaml             # Base configuration
│   ├── experiments/                 # Experiment-specific configs
│   │   ├── cell_masking.yaml       # Cell-level masking experiments
│   │   ├── attribute_masking.yaml  # Attribute-level masking
│   │   └── scheduled_masking.yaml  # Scheduled masking probability
│   └── model_configs/               # Model configurations
│       └── llada2_mini.yaml        # LLaDA2.0-mini config
├── src/                             # Source code
│   ├── __init__.py
│   ├── data/                        # Data processing
│   │   ├── __init__.py
│   │   ├── minigrid_dataset.py     # MiniGrid data loader
│   │   ├── babyai_dataset.py       # BabyAI data loader
│   │   ├── trajectory_processor.py # Trajectory processing
│   │   └── data_utils.py           # Utility functions
│   ├── masking/                     # Masking strategies
│   │   ├── __init__.py
│   │   ├── mask_sampler.py         # Core mask sampling logic
│   │   ├── cell_masker.py          # Cell-level masking
│   │   ├── attribute_masker.py     # Attribute-level masking
│   │   └── scheduled_masker.py     # Scheduled masking
│   ├── models/                      # Model wrappers
│   │   ├── __init__.py
│   │   ├── llada_wrapper.py        # LLaDA2.0 wrapper
│   │   └── model_utils.py          # Model utilities
│   ├── training/                    # Training logic
│   │   ├── __init__.py
│   │   ├── trainer.py              # Main training loop
│   │   ├── loss.py                 # Loss functions
│   │   └── callbacks.py            # Training callbacks
│   └── evaluation/                  # Evaluation
│       ├── __init__.py
│       ├── evaluator.py            # Main evaluation logic
│       ├── metrics.py              # Metrics computation
│       └── visualizer.py           # Visualization tools
├── scripts/                         # Utility scripts
│   ├── setup_dfactory.sh           # Setup dFactory
│   ├── download_model.sh           # Download LLaDA2.0
│   ├── generate_trajectories.py    # Generate training data
│   ├── train.py                    # Main training script
│   └── evaluate.py                 # Main evaluation script
├── experiments/                     # Experiment tracking
│   └── .gitkeep
├── data/                           # Data directory
│   ├── raw/                        # Raw trajectories
│   ├── processed/                  # Processed data
│   └── .gitkeep
├── outputs/                        # Output directory
│   ├── checkpoints/                # Model checkpoints
│   ├── logs/                       # Training logs
│   └── results/                    # Evaluation results
└── tests/                          # Unit tests
    ├── __init__.py
    ├── test_masking.py
    ├── test_data.py
    └── test_training.py
```

## Quick Start

### 1. Environment Setup

```bash
# Clone dFactory repository
git clone https://github.com/inclusionAI/dFactory.git --recursive
cd dFactory/VeOmni

# Install dependencies using uv (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --extra gpu
source .venv/bin/activate
cd ../..

# Clone this project
git clone <this-repo>
cd any-order-training

# Install project dependencies
pip install -e .
```

### 2. Download and Setup Model

```bash
# Download LLaDA2.0-mini-preview
bash scripts/download_model.sh

# Merge model weights for training
bash scripts/setup_model.sh
```

### 3. Generate Training Data

```bash
# Generate MiniGrid trajectories
python scripts/generate_trajectories.py \
    --env BabyAI-GoToRedBall-v0 \
    --num_episodes 1000 \
    --output_dir data/raw/babyai_goto

# Process trajectories for training
python scripts/process_data.py \
    --input_dir data/raw/babyai_goto \
    --output_dir data/processed/babyai_goto \
    --masking_strategy cell
```

### 4. Run Training

```bash
# Cell-level masking experiment
python scripts/train.py \
    --config configs/experiments/cell_masking.yaml \
    --output_dir outputs/cell_masking_exp1

# Attribute-level masking experiment
python scripts/train.py \
    --config configs/experiments/attribute_masking.yaml \
    --output_dir outputs/attribute_masking_exp1
```

### 5. Evaluate

```bash
# Evaluate trained model
python scripts/evaluate.py \
    --checkpoint outputs/cell_masking_exp1/checkpoints/best \
    --env BabyAI-GoToRedBall-v0 \
    --num_episodes 100
```

## Masking Strategies

### Cell-Level Masking
Masks entire cells (all 3 attributes together):
```
[cell_i_obj, cell_i_color, cell_i_state] → [MASK]
```

### Attribute-Level Masking
Masks individual attributes:
```
[door, red, MASK] → predict state (closed/open)
```

### Scheduled Masking
Gradually increases masking probability during training.

## Configuration

All experiments are configured via YAML files in `configs/experiments/`. Key parameters:

- `masking_prob`: Probability of masking (0.15-0.50)
- `masking_strategy`: 'cell' or 'attribute'
- `mask_schedule`: Optional probability scheduling
- `batch_size`: Training batch size
- `learning_rate`: Learning rate
- `num_epochs`: Number of training epochs

## Evaluation Metrics

- **World Model NLL**: Negative log-likelihood of observations
- **Action Accuracy**: Accuracy of predicted actions
- **Task Success Rate**: Success rate on held-out tasks
- **Partial Observability Robustness**: Performance under occlusions

## Citation

```bibtex
@misc{anyorder2025,
  title={Any-Order Masked Training for Trajectory-Level Learning in LLM-Based Agents},
  author={Your Name},
  year={2025}
}
```

## License

Apache 2.0
