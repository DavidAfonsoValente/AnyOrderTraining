# Testing Guide

This document provides step-by-step instructions for testing the Any-Order Masked Training implementation.

## Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended, but CPU works)
- 16GB+ RAM
- 20GB+ disk space

## Installation

### Option 1: Quick Start (Recommended for Testing)

```bash
# Clone repository
cd any-order-training

# Run quick start script
bash quickstart.sh
```

This will:
1. Install all dependencies
2. Generate test data (200 trajectories)
3. Run a short training (3 epochs) to verify the pipeline

### Option 2: Manual Setup

```bash
# Install dependencies
pip install -e .
pip install minigrid gymnasium

# Create directories
mkdir -p data/raw data/processed outputs experiments

# Generate data
python scripts/generate_trajectories.py \
    --env BabyAI-GoToRedBall-v0 \
    --num_episodes 200 \
    --output_dir data/raw
```

## Testing the Pipeline

### 1. Unit Tests (Basic Functionality)

```bash
# Test data loading
python -c "
from src.data.minigrid_dataset import BabyAIDataset
dataset = BabyAIDataset('data/raw/BabyAI-GoToRedBall-v0', split='train')
print(f'Dataset loaded: {len(dataset)} trajectories')
sample = dataset[0]
print(f'Sample keys: {list(sample.keys())}')
print(f'Observation shape: {sample[\"observations\"].shape}')
print('✓ Data loading works!')
"

# Test masking
python -c "
import torch
from src.masking.cell_masker import CellMasker

masker = CellMasker(mask_prob=0.3)
obs = torch.randint(0, 10, (2, 10, 7, 7, 3))
act = torch.randint(0, 7, (2, 10))
mask = torch.ones(2, 10)

result = masker.sample_mask(obs, act, mask)
print(f'Mask stats: {result[\"mask_stats\"]}')
print('✓ Masking works!')
"
```

### 2. Integration Test (Mini Training Run)

```bash
# Run 1 epoch of training
python scripts/train.py \
    --config configs/experiments/cell_masking.yaml \
    --training.num_epochs 1 \
    --data.batch_size 4 \
    --output_dir outputs/integration_test

# Check outputs
ls outputs/integration_test/checkpoints/
ls outputs/integration_test/logs/

echo "✓ Training pipeline works!"
```

### 3. Data Generation Test

```bash
# Generate small dataset
python scripts/generate_trajectories.py \
    --env BabyAI-GoToRedBall-v0 \
    --num_episodes 50 \
    --max_steps 30 \
    --output_dir data/test

# Verify output
echo "Train trajectories: $(ls data/test/BabyAI-GoToRedBall-v0/train/*.json | wc -l)"
echo "Val trajectories: $(ls data/test/BabyAI-GoToRedBall-v0/val/*.json | wc -l)"
echo "Test trajectories: $(ls data/test/BabyAI-GoToRedBall-v0/test/*.json | wc -l)"

echo "✓ Data generation works!"
```

### 4. Masking Strategy Test

Test all three masking strategies:

```bash
# Cell-level masking
python -c "
from src.masking.cell_masker import CellMasker
import torch

masker = CellMasker(mask_prob=0.3)
obs = torch.randint(0, 10, (2, 5, 7, 7, 3))
act = torch.randint(0, 7, (2, 5))
mask = torch.ones(2, 5)

result = masker.sample_mask(obs, act, mask)
print('Cell-level masking:')
print(f'  Masked cells: {result[\"mask_stats\"][\"masked_cells\"]}')
print(f'  Mask ratio: {result[\"mask_stats\"][\"cell_mask_ratio\"]:.3f}')
print('✓ Cell masking works!')
"

# Attribute-level masking
python -c "
from src.masking.attribute_masker import AttributeMasker
import torch

masker = AttributeMasker(mask_prob=0.3)
obs = torch.randint(0, 10, (2, 5, 7, 7, 3))
act = torch.randint(0, 7, (2, 5))
mask = torch.ones(2, 5)

result = masker.sample_mask(obs, act, mask)
print('Attribute-level masking:')
print(f'  Object mask ratio: {result[\"mask_stats\"][\"object_mask_ratio\"]:.3f}')
print(f'  Color mask ratio: {result[\"mask_stats\"][\"color_mask_ratio\"]:.3f}')
print(f'  State mask ratio: {result[\"mask_stats\"][\"state_mask_ratio\"]:.3f}')
print('✓ Attribute masking works!')
"

# Scheduled masking
python -c "
from src.masking.cell_masker import ScheduledCellMasker
import torch

masker = ScheduledCellMasker(
    start_prob=0.15,
    end_prob=0.50,
    schedule_steps=100,
    schedule_type='linear'
)

print('Scheduled masking:')
for step in [0, 25, 50, 75, 100]:
    masker.update_step(step)
    prob = masker.get_current_mask_prob()
    print(f'  Step {step:3d}: mask_prob = {prob:.3f}')

print('✓ Scheduled masking works!')
"
```

### 5. Full Experiment Test

Run a complete but small-scale experiment:

```bash
# Generate data (if not already done)
python scripts/generate_trajectories.py \
    --env BabyAI-GoToRedBall-v0 \
    --num_episodes 500 \
    --output_dir data/raw

# Train for 10 epochs
python scripts/train.py \
    --config configs/experiments/cell_masking.yaml \
    --training.num_epochs 10 \
    --data.batch_size 8 \
    --output_dir outputs/full_test

# Evaluate
python scripts/evaluate.py \
    --checkpoint outputs/full_test/checkpoints/best.pt \
    --metric all \
    --data_dir data/raw/BabyAI-GoToRedBall-v0 \
    --env BabyAI-GoToRedBall-v0

# Check results
cat outputs/full_test/evaluation_results.json

echo "✓ Full experiment works!"
```

## Expected Outputs

### Successful Data Generation

```
Collecting 200 trajectories from BabyAI-GoToRedBall-v0...
Policy: random, Max steps: 100
Saving to: data/raw/BabyAI-GoToRedBall-v0/train
100%|███████████████████| 140/140 [00:30<00:00,  4.62it/s]

Collection complete!
Success rate: 45% (63/140)
Average trajectory length: 42.3
Trajectories saved to: data/raw/BabyAI-GoToRedBall-v0/train
```

### Successful Training

```
Loaded 140 trajectories from train split
Loaded 30 trajectories from val split
Train dataset: 140 trajectories
Val dataset: 30 trajectories

Creating masker...
Masking strategy: cell
Masking probability: 0.3

Creating model...
Model created
Using device: cuda

Starting Training
================================================================================

Epoch 1/10:   0%|          | 0/18 [00:00<?, ?it/s]
Epoch 1/10: 100%|██████████| 18/18 [00:05<00:00,  3.2it/s, loss=2.456, lr=1e-05]

Evaluating:  100%|██████████| 4/4 [00:01<00:00,  3.8it/s]

Epoch 1:
  Train Loss: 2.4563
  Val Loss: 2.3891
```

### Successful Evaluation

```
Loading checkpoint from outputs/full_test/checkpoints/best.pt
Using device: cuda
Evaluating metric: all

=== Computing World Model NLL ===
World Model NLL: 1.8234

=== Computing Accuracy Metrics ===
Accuracy Metrics:
  observation_accuracy: 0.7234
  object_accuracy: 0.8123
  color_accuracy: 0.7845
  state_accuracy: 0.5734
  action_accuracy: 0.6123

=== Evaluating Task Success on BabyAI-GoToRedBall-v0 ===
Task Success Metrics:
  success_rate: 0.4200
  avg_reward: 0.4200
  avg_episode_length: 48.23
  std_episode_length: 12.45

=== Evaluating Robustness on BabyAI-GoToRedBall-v0 ===
Robustness Metrics:
  success_rate: 0.3100
  avg_reward: 0.3100
  occlusion_prob: 0.3

Results saved to outputs/full_test/evaluation_results.json

=== Evaluation Complete ===
```

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**
   ```bash
   # Make sure you installed the package
   pip install -e .
   ```

2. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   python scripts/train.py --config ... --data.batch_size 2
   ```

3. **No trajectories generated**
   ```bash
   # Check if environment name is correct
   python -c "import gymnasium as gym; gym.make('BabyAI-GoToRedBall-v0')"
   ```

4. **Training loss is NaN**
   ```bash
   # Reduce learning rate
   python scripts/train.py --config ... --training.learning_rate 1e-6
   ```

### Verification Checklist

- [ ] Dependencies installed
- [ ] Data generated successfully
- [ ] Training runs without errors
- [ ] Loss decreases over epochs
- [ ] Checkpoints saved
- [ ] Evaluation completes
- [ ] Results saved to JSON

## Next Steps

After verifying the pipeline works:

1. **Generate full dataset** (1000+ trajectories)
   ```bash
   python scripts/generate_trajectories.py \
       --env BabyAI-GoToRedBall-v0 \
       --num_episodes 1200 \
       --output_dir data/raw
   ```

2. **Run ablation studies**
   ```bash
   bash scripts/run_ablations.sh
   ```

3. **Analyze results**
   ```bash
   cat outputs/ablations/RESULTS_SUMMARY.md
   ```

## Performance Benchmarks

Expected performance on a modest GPU (RTX 3080):

- Data generation: ~5 episodes/second
- Training: ~30 batches/second (batch_size=8)
- Evaluation: ~50 episodes/second

For 10 epochs on 1000 trajectories:
- Training time: ~45 minutes
- Evaluation time: ~5 minutes

## Contact

If you encounter issues not covered here, please check:
- README.md for project overview
- EXPERIMENTS.md for experiment details
- GitHub issues (if repository is public)
