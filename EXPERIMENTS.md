# Experiments and Testing Guide

This document outlines the experimental setup and testing procedures for the Any-Order Masked Training project.

## Phase 1: Small-Scale Ablations on MiniGrid/BabyAI

### Experiment 1: Cell-Level Masking with Different Probabilities

**Objective**: Determine optimal masking probability for cell-level masking.

**Setup**:
- Environment: `BabyAI-GoToRedBall-v0`
- Dataset: 1000 training, 100 validation, 100 test trajectories
- Masking strategy: Cell-level (all 3 attributes masked together)
- Masking probabilities to test: [0.15, 0.30, 0.50]

**Commands**:
```bash
# Generate data (only once)
python scripts/generate_trajectories.py \
    --env BabyAI-GoToRedBall-v0 \
    --num_episodes 1200 \
    --output_dir data/raw \
    --policy random

# Train with p=0.15
python scripts/train.py \
    --config configs/experiments/cell_masking.yaml \
    --masking.mask_prob 0.15 \
    --output_dir outputs/cell_p015

# Train with p=0.30
python scripts/train.py \
    --config configs/experiments/cell_masking.yaml \
    --masking.mask_prob 0.30 \
    --output_dir outputs/cell_p030

# Train with p=0.50
python scripts/train.py \
    --config configs/experiments/cell_masking.yaml \
    --masking.mask_prob 0.50 \
    --output_dir outputs/cell_p050
```

**Metrics to Compare**:
- World model NLL
- Observation reconstruction accuracy (overall and per-attribute)
- Action prediction accuracy
- Training convergence speed

**Expected Results**:
- Higher masking probability → harder task
- Sweet spot likely around 0.30-0.40
- Too high (>0.60) may prevent learning

---

### Experiment 2: Cell vs Attribute Level Masking

**Objective**: Compare coarse (cell-level) vs fine-grained (attribute-level) masking.

**Setup**:
- Environment: `BabyAI-GoToRedBall-v0`
- Dataset: Same as Experiment 1
- Masking probability: 0.30 (fixed)
- Strategies: Cell-level vs Attribute-level

**Commands**:
```bash
# Cell-level masking
python scripts/train.py \
    --config configs/experiments/cell_masking.yaml \
    --output_dir outputs/cell_vs_attr/cell

# Attribute-level masking
python scripts/train.py \
    --config configs/experiments/attribute_masking.yaml \
    --output_dir outputs/cell_vs_attr/attribute
```

**Metrics to Compare**:
- Reconstruction accuracy per attribute (object, color, state)
- Dynamics learning quality (measured by next-state prediction)
- Downstream task performance

**Hypothesis**:
- Attribute-level should learn finer-grained relationships
- Cell-level should be more stable/easier to train
- Attribute-level may excel at understanding affordances and dynamics

---

### Experiment 3: Scheduled Masking (Curriculum Learning)

**Objective**: Test if gradual increase in masking difficulty improves learning.

**Setup**:
- Environment: `BabyAI-GoToRedBall-v0`
- Dataset: Same as Experiment 1
- Schedule: Linear 0.15 → 0.50 over 10k steps
- Baseline: Fixed 0.30

**Commands**:
```bash
# Scheduled masking
python scripts/train.py \
    --config configs/experiments/scheduled_masking.yaml \
    --output_dir outputs/scheduled

# Fixed baseline (for comparison)
python scripts/train.py \
    --config configs/experiments/cell_masking.yaml \
    --output_dir outputs/fixed_baseline
```

**Metrics to Compare**:
- Training loss curves
- Final performance
- Sample efficiency (performance vs training steps)

**Hypothesis**:
- Scheduled masking should converge faster
- May achieve better final performance
- Particularly beneficial for harder tasks

---

### Experiment 4: Multi-Environment Generalization

**Objective**: Test generalization across different BabyAI tasks.

**Setup**:
- Train environments: `BabyAI-GoToRedBall-v0`, `BabyAI-Pickup-v0`
- Test environments: `BabyAI-Open-v0`, `BabyAI-GoToObj-v0`
- Masking strategy: Best from Experiments 1-3

**Commands**:
```bash
# Generate data for multiple environments
for env in BabyAI-GoToRedBall-v0 BabyAI-Pickup-v0 BabyAI-Open-v0; do
    python scripts/generate_trajectories.py \
        --env $env \
        --num_episodes 1000 \
        --output_dir data/raw
done

# Train on combined data
python scripts/train.py \
    --config configs/experiments/multi_env.yaml \
    --output_dir outputs/multi_env
```

**Metrics**:
- Zero-shot transfer performance
- Fine-tuning sample efficiency

---

## Phase 2: Full Benchmarks (Future Work)

### WebArena Benchmark
- Web navigation tasks
- Longer horizons
- Richer observation space

### ToolBench Benchmark
- Tool use and API calling
- Multi-step reasoning
- Real-world applicability

---

## Testing Checklist

### Before Running Experiments

- [ ] Environment setup complete
- [ ] Data generated and validated
- [ ] Config files reviewed
- [ ] Output directories created
- [ ] Logging configured (wandb/tensorboard)

### During Training

- [ ] Monitor loss curves
- [ ] Check masking statistics
- [ ] Verify GPU utilization
- [ ] Watch for NaN/Inf values
- [ ] Save checkpoints regularly

### After Training

- [ ] Evaluate on validation set
- [ ] Evaluate on test set
- [ ] Compare against baselines
- [ ] Visualize predictions
- [ ] Document findings

---

## Quick Start Testing

For quick testing of the pipeline (no full training):

```bash
# Generate small dataset (100 episodes)
python scripts/generate_trajectories.py \
    --env BabyAI-GoToRedBall-v0 \
    --num_episodes 120 \
    --output_dir data/raw/test

# Run training for 1 epoch
python scripts/train.py \
    --config configs/experiments/cell_masking.yaml \
    --training.num_epochs 1 \
    --data.batch_size 4 \
    --output_dir outputs/test_run

# Check outputs
ls outputs/test_run/checkpoints/
ls outputs/test_run/logs/
```

---

## Evaluation Protocol

### 1. World Model Quality
```bash
python scripts/evaluate.py \
    --checkpoint outputs/cell_p030/checkpoints/best.pt \
    --metric world_model_nll \
    --data_dir data/processed/BabyAI-GoToRedBall-v0
```

### 2. Task Success Rate
```bash
python scripts/evaluate.py \
    --checkpoint outputs/cell_p030/checkpoints/best.pt \
    --metric task_success \
    --env BabyAI-GoToRedBall-v0 \
    --num_episodes 100
```

### 3. Partial Observability Robustness
```bash
python scripts/evaluate.py \
    --checkpoint outputs/cell_p030/checkpoints/best.pt \
    --metric robustness \
    --env BabyAI-GoToRedBall-v0 \
    --occlusion_prob 0.3
```

---

## Analysis and Visualization

After running experiments, analyze results with:

```bash
# Compare experiments
python scripts/compare_experiments.py \
    --exp_dirs outputs/cell_p015 outputs/cell_p030 outputs/cell_p050

# Generate plots
python scripts/plot_results.py \
    --metrics_file outputs/cell_p030/metrics.json \
    --output_dir outputs/cell_p030/plots

# Create report
python scripts/generate_report.py \
    --exp_dir outputs/cell_p030 \
    --output outputs/cell_p030/report.md
```

---

## Troubleshooting

### Common Issues

1. **Out of Memory (OOM)**
   - Reduce batch size: `--data.batch_size 4`
   - Reduce sequence length: `--data.max_trajectory_length 64`
   - Use gradient accumulation

2. **Training Instability**
   - Lower learning rate: `--training.learning_rate 1e-6`
   - Increase warmup steps
   - Check for NaN in gradients

3. **Poor Performance**
   - Verify data quality
   - Check masking statistics (should match config)
   - Try different masking strategies
   - Reduce masking probability

4. **Slow Training**
   - Increase batch size (if memory allows)
   - Use mixed precision training (bf16/fp16)
   - Reduce number of workers if CPU-bound
   - Check data loading pipeline

---

## Expected Timeline

- **Week 1**: Setup, data generation, pipeline testing
- **Week 2-3**: Experiments 1-3 (masking ablations)
- **Week 4**: Experiment 4 (generalization)
- **Week 5-6**: Analysis, visualization, documentation

---

## Success Criteria

### Minimum Viable Results
- Training completes without errors
- Loss decreases over time
- Observation accuracy > 60%
- Action accuracy > 40%

### Good Results
- Observation accuracy > 80%
- Action accuracy > 60%
- Successful generalization to new tasks
- Clear benefit of any-order masking over baselines

### Excellent Results
- Observation accuracy > 90%
- Action accuracy > 75%
- Zero-shot transfer to unseen tasks
- Outperforms single-order baselines significantly
