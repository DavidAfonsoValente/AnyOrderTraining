# Any-Order Masked Training - Quick Start

Welcome! This project implements Any-Order Masked Training for trajectory-level learning in LLM-based agents.

## 🚀 Get Started in 5 Minutes

```bash
# 1. Run quickstart script
bash quickstart.sh

# That's it! The script will:
#   - Install dependencies
#   - Generate test data
#   - Run a short training
#   - Verify everything works
```

## 📁 What You Have

```
any-order-training/
├── README.md                    # Full project documentation
├── PROJECT_STATUS.md            # Current status & next steps  ⭐ START HERE
├── EXPERIMENTS.md               # Ablation study plans
├── TESTING.md                   # Testing procedures
├── quickstart.sh                # One-command setup
├── configs/                     # Experiment configurations
│   ├── base_config.yaml         # Base configuration
│   └── experiments/             # Experiment-specific configs
│       ├── cell_masking.yaml
│       ├── attribute_masking.yaml
│       └── scheduled_masking.yaml
├── src/                         # Source code
│   ├── data/                    # Data loading & processing
│   ├── masking/                 # Masking strategies (core contribution!)
│   ├── models/                  # Model wrappers
│   ├── training/                # Training logic
│   └── evaluation/              # Evaluation metrics
├── scripts/                     # Executable scripts
│   ├── generate_trajectories.py # Generate training data
│   ├── train.py                 # Main training script
│   ├── evaluate.py              # Evaluation script
│   └── run_ablations.sh         # Run full ablation study
└── data/                        # Data directory (created by scripts)
```

## 🎯 Current Status

✅ **Ready to Use:**
- Complete data pipeline
- All 3 masking strategies (cell, attribute, scheduled)
- Training infrastructure
- Evaluation suite
- Ablation experiment scripts

🚧 **Next Step:**
- Integrate with real LLaDA2.0 model (currently using mock model)

## 📖 Key Documents

1. **PROJECT_STATUS.md** ⭐ - Read this first!
   - Shows what's done vs what's needed
   - Clear integration checklist
   - Step-by-step guide for LLaDA2.0 integration

2. **EXPERIMENTS.md** - Ablation study plans
   - 4 detailed experiments
   - Expected results
   - Analysis procedures

3. **TESTING.md** - Verification guide
   - Unit tests
   - Integration tests
   - Troubleshooting

## 🔬 Quick Experiment

```bash
# Generate data (1200 trajectories, ~5 min)
python scripts/generate_trajectories.py \
    --env BabyAI-GoToRedBall-v0 \
    --num_episodes 1200 \
    --output_dir data/raw

# Train with cell-level masking (30% probability)
python scripts/train.py \
    --config configs/experiments/cell_masking.yaml \
    --output_dir outputs/exp1

# Evaluate
python scripts/evaluate.py \
    --checkpoint outputs/exp1/checkpoints/best.pt \
    --metric all
```

## 🎨 Masking Strategies Implemented

### 1. Cell-Level Masking
```
[cell_obj, cell_color, cell_state] → [MASK, MASK, MASK]
```
- Masks all 3 attributes together
- Teaches spatial reasoning & object permanence
- Config: `configs/experiments/cell_masking.yaml`

### 2. Attribute-Level Masking
```
[door, red, MASK] → predict state (open/closed)
```
- Masks individual attributes
- Teaches dynamics, affordances, causal structure
- Config: `configs/experiments/attribute_masking.yaml`

### 3. Scheduled Masking
```
Probability: 0.15 → 0.30 → 0.50 (gradually increases)
```
- Curriculum learning approach
- Config: `configs/experiments/scheduled_masking.yaml`

## 🏃 Run Full Ablation Study

```bash
# Runs all experiments automatically
bash scripts/run_ablations.sh

# Results will be in:
#   outputs/ablations/RESULTS_SUMMARY.md
```

## 💡 What Makes This Special

**Any-Order Masking**: 
- Traditional SFT: Fixed order (past → predict next action)
- This project: Random masks each epoch (any element can be masked)
- Benefits: Learns richer representations, better generalization

**Key Innovation**:
- Single-pass reconstruction (no multi-step diffusion)
- Works with masked DLMs like LLaDA2.0
- Subsumes standard SFT as special case

## 🛠️ Integration with LLaDA2.0

See `PROJECT_STATUS.md` for detailed integration guide.

Quick summary:
1. Download LLaDA2.0 model
2. Replace mock model in `scripts/train.py`
3. Create tokenization layer
4. Run experiments!

## 📊 Expected Results

With mock model (testing only):
- ✅ Pipeline works end-to-end
- ✅ Training converges
- ✅ Metrics computed

With real LLaDA2.0:
- 🎯 Observation accuracy > 80%
- 🎯 Action accuracy > 60%
- 🎯 Task success > 40%

## 🆘 Need Help?

1. **Something not working?** → Check `TESTING.md`
2. **Want to run experiments?** → See `EXPERIMENTS.md`
3. **Ready to integrate LLaDA2.0?** → Read `PROJECT_STATUS.md`
4. **General questions?** → See `README.md`

## 📝 Citation

```bibtex
@misc{anyorder2025,
  title={Any-Order Masked Training for Trajectory-Level Learning in LLM-Based Agents},
  author={Your Name},
  year={2025}
}
```

## 🎉 You're Ready!

The project is fully implemented and tested. Next step: integrate with LLaDA2.0 and run the experiments!

**Start here:** `PROJECT_STATUS.md` → Integration Checklist
