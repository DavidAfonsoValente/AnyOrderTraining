# Any-Order Masked Training - Document Index

## 📖 Start Here

### For Quick Overview
👉 **[QUICKSTART_README.md](QUICKSTART_README.md)** - One-page project overview

### For Immediate Action
👉 **[PROJECT_STATUS.md](PROJECT_STATUS.md)** - What's done, what's next, integration checklist

### For Understanding the Project
👉 **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Comprehensive summary of everything

## 📚 Full Documentation

### Core Documentation
- **[README.md](README.md)** - Full project documentation with architecture details
- **[EXPERIMENTS.md](EXPERIMENTS.md)** - Detailed ablation study plans (4 experiments)
- **[TESTING.md](TESTING.md)** - Testing procedures and verification
- **[PROJECT_STATUS.md](PROJECT_STATUS.md)** - Current status and next steps

### Quick Reference
- **[QUICKSTART_README.md](QUICKSTART_README.md)** - Get started in 5 minutes
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - What's implemented and how to use it
- **[DIRECTORY_STRUCTURE.txt](DIRECTORY_STRUCTURE.txt)** - Complete file listing

## 🎯 Reading Guide by Goal

### "I want to understand what this is"
1. Read [QUICKSTART_README.md](QUICKSTART_README.md) (5 min)
2. Skim [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) (10 min)

### "I want to test the pipeline"
1. Read [TESTING.md](TESTING.md) (15 min)
2. Run `bash quickstart.sh` (5 min)
3. Check outputs work

### "I want to integrate with LLaDA2.0"
1. Read [PROJECT_STATUS.md](PROJECT_STATUS.md) → Integration Checklist (30 min)
2. Follow step-by-step guide
3. Test with small run

### "I want to run experiments"
1. Read [EXPERIMENTS.md](EXPERIMENTS.md) (20 min)
2. Generate data: `python scripts/generate_trajectories.py`
3. Run: `bash scripts/run_ablations.sh`

### "I want to understand the code"
1. Read [README.md](README.md) → Architecture section
2. Look at code in `src/masking/` (core contribution)
3. Review `src/training/trainer.py` (training loop)

## 📁 Directory Guide

```
any-order-training/
├── 📄 Documentation (you are here)
│   ├── INDEX.md                      # This file - navigation guide
│   ├── QUICKSTART_README.md          # Start here!
│   ├── PROJECT_STATUS.md             # Current status & next steps
│   ├── IMPLEMENTATION_SUMMARY.md     # Complete summary
│   ├── README.md                     # Full documentation
│   ├── EXPERIMENTS.md                # Experiment plans
│   └── TESTING.md                    # Testing guide
│
├── ⚙️ Configuration
│   └── configs/
│       ├── base_config.yaml          # Base settings
│       └── experiments/              # 3 experiment configs
│
├── 💻 Source Code
│   └── src/
│       ├── data/                     # Data loading
│       ├── masking/                  # ⭐ Core contribution
│       ├── models/                   # Model wrappers
│       ├── training/                 # Training loop
│       └── evaluation/               # Metrics
│
├── 🔧 Scripts
│   └── scripts/
│       ├── generate_trajectories.py  # Data collection
│       ├── train.py                  # Training
│       ├── evaluate.py               # Evaluation
│       └── run_ablations.sh          # Automated experiments
│
└── 🚀 Quick Start
    ├── quickstart.sh                 # One-command setup
    ├── requirements.txt              # Dependencies
    └── setup.py                      # Package installer
```

## 🎨 Key Concepts

### Any-Order Masking
- Different masks each epoch
- Single-pass reconstruction
- No multi-step diffusion
- See: [README.md](README.md) → Motivation

### Masking Strategies
1. **Cell-level** - Mask whole cells
2. **Attribute-level** - Mask individual attributes
3. **Scheduled** - Curriculum learning
- See: [QUICKSTART_README.md](QUICKSTART_README.md) → Masking Strategies

### Experiments
1. Masking probability ablation
2. Cell vs attribute comparison
3. Scheduled vs fixed masking
4. Multi-environment generalization
- See: [EXPERIMENTS.md](EXPERIMENTS.md)

## 🔍 Find Information Fast

| I want to... | Read this... |
|--------------|-------------|
| Get started in 5 min | [QUICKSTART_README.md](QUICKSTART_README.md) |
| Understand the project | [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) |
| Integrate with LLaDA2.0 | [PROJECT_STATUS.md](PROJECT_STATUS.md) |
| Run experiments | [EXPERIMENTS.md](EXPERIMENTS.md) |
| Test the code | [TESTING.md](TESTING.md) |
| See full details | [README.md](README.md) |

## 📊 Project Status

- ✅ **Implementation**: 100% Complete
- 🔧 **Integration**: Waiting for LLaDA2.0 (2-3 hours)
- 📅 **Experiments**: Ready to run (after integration)
- 📈 **Results**: TBD (2-3 weeks)

## 🎯 Next Steps

1. **Read** [PROJECT_STATUS.md](PROJECT_STATUS.md)
2. **Test** with `bash quickstart.sh`
3. **Integrate** LLaDA2.0 (follow checklist)
4. **Experiment** with `bash scripts/run_ablations.sh`

## 💡 Quick Commands

```bash
# Test pipeline
bash quickstart.sh

# Generate data
python scripts/generate_trajectories.py --env BabyAI-GoToRedBall-v0 --num_episodes 1200

# Train
python scripts/train.py --config configs/experiments/cell_masking.yaml

# Evaluate
python scripts/evaluate.py --checkpoint outputs/exp/checkpoints/best.pt --metric all

# Run all experiments
bash scripts/run_ablations.sh
```

## 📦 Package Contents

This archive contains:
- ✅ Complete source code (all modules)
- ✅ 3 experiment configurations
- ✅ Training and evaluation scripts
- ✅ Data generation tools
- ✅ Automated ablation pipeline
- ✅ Comprehensive documentation (7 docs)
- ✅ Quick start script
- ✅ Testing suite

**Total**: Production-ready research codebase

## 🚀 You're Ready!

Everything you need is here. Start with [QUICKSTART_README.md](QUICKSTART_README.md) and you'll be running experiments within the day!

---

**Project**: Any-Order Masked Training for Trajectory-Level Learning  
**Status**: Ready for integration and experimentation  
**Last Updated**: February 7, 2026
