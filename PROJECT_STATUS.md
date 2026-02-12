# Project Status: Any-Order Masked Training

## Overview

This document provides the current status of the implementation and clear next steps for integration with LLaDA2.0 and dFactory.

## ✅ Completed Components

### 1. Data Pipeline
- ✅ MiniGrid/BabyAI trajectory collection
- ✅ Dataset loading and batching
- ✅ Trajectory preprocessing
- ✅ Train/val/test splits

### 2. Masking Strategies
- ✅ Cell-level masking (masks all 3 attributes together)
- ✅ Attribute-level masking (masks individual attributes)
- ✅ Scheduled masking (curriculum learning)
- ✅ Random mask sampling (baseline any-order)

### 3. Training Infrastructure
- ✅ Training loop with checkpointing
- ✅ Loss functions (masked reconstruction)
- ✅ Optimizer and scheduler setup
- ✅ Logging (Tensorboard + Wandb)

### 4. Evaluation
- ✅ World model NLL computation
- ✅ Accuracy metrics (observation + action)
- ✅ Task success rate evaluation
- ✅ Partial observability robustness testing

### 5. Experiment Management
- ✅ YAML configuration system
- ✅ Multiple experiment configs
- ✅ Automated ablation scripts
- ✅ Results comparison tools

### 6. Documentation
- ✅ README with quickstart
- ✅ EXPERIMENTS.md with detailed ablation plans
- ✅ TESTING.md with verification procedures
- ✅ Code comments and docstrings

## 🚧 TODO: Critical Integration Tasks

### Priority 1: LLaDA2.0 Model Integration

**Current State**: Using a mock model for pipeline testing

**What Needs to Be Done**:

1. **Replace Mock Model** in `scripts/train.py`:
   ```python
   # CURRENT (line ~50):
   def create_model(config: dict):
       class MockModel(nn.Module):
           # ... mock implementation
       return MockModel()
   
   # REPLACE WITH:
   def create_model(config: dict):
       from transformers import AutoModelForCausalLM
       
       model_path = config['model']['model_path']
       model = AutoModelForCausalLM.from_pretrained(
           model_path,
           trust_remote_code=True,
           torch_dtype=torch.bfloat16,
       )
       return model
   ```

2. **Update Forward Pass** in `src/training/trainer.py`:
   ```python
   # CURRENT (line ~180):
   outputs = self.model(
       observations=masked_obs,
       actions=masked_act,
       attention_mask=attention_mask,
   )
   
   # REPLACE WITH LLaDA2.0 API:
   # Need to convert obs/act to tokens first, then call model
   # See LLaDA2.0 documentation for exact API
   ```

3. **Add Tokenization Layer**:
   - Create `src/models/llada_tokenizer.py`
   - Convert MiniGrid obs/act to token sequences
   - Handle BabyAI-specific vocabulary

### Priority 2: dFactory Integration

**Location**: Need to adapt training to use dFactory's training framework

**Files to Modify**:
1. Copy this project structure into dFactory's `tasks/` directory
2. Create `tasks/train_any_order_babyai.py` similar to `tasks/train_llada2_bd.py`
3. Update configs to match dFactory's format
4. Use dFactory's data loading utilities if beneficial

**Steps**:
```bash
# 1. Clone dFactory (if not already done)
git clone https://github.com/inclusionAI/dFactory.git --recursive

# 2. Copy any-order-training into dFactory
cp -r any-order-training dFactory/tasks/any_order_training

# 3. Create dFactory-compatible training script
# See dFactory/tasks/train_llada2_bd.py as template

# 4. Integrate with dFactory config system
# See dFactory/configs/sft/llada2_mini_bd_sft.yaml
```

### Priority 3: Model Download and Setup

**What Needs to Be Done**:

1. **Download LLaDA2.0-mini**:
   ```bash
   python ./scripts/download_hf_model.py \
       --repo_id inclusionAI/LLaDA2.0-mini-preview \
       --local_dir /path/to/llada2_mini
   ```

2. **Convert to Merged Format**:
   ```bash
   python scripts/moe_convertor.py \
       --input-path /path/to/llada2_mini \
       --output-path /path/to/llada2_mini_merged \
       --mode merge
   ```

3. **Update Config**:
   ```yaml
   # configs/base_config.yaml
   model:
       model_path: "/path/to/llada2_mini_merged"
   ```

## 📋 Integration Checklist

### Step 1: Setup dFactory
- [ ] Clone dFactory repository
- [ ] Install VeOmni dependencies
- [ ] Download LLaDA2.0-mini model
- [ ] Convert model to merged format
- [ ] Verify model loads correctly

### Step 2: Integrate Any-Order Training
- [ ] Copy project into dFactory/tasks/
- [ ] Create dFactory-compatible training script
- [ ] Adapt configs to dFactory format
- [ ] Update data loading to use dFactory utilities (optional)

### Step 3: Replace Mock Components
- [ ] Implement real LLaDA2.0 model loading
- [ ] Create tokenization layer for MiniGrid observations
- [ ] Update forward pass to use LLaDA2.0 API
- [ ] Verify masked reconstruction works with real model

### Step 4: Test Integration
- [ ] Run quickstart.sh with real model
- [ ] Verify training runs without errors
- [ ] Check that loss decreases
- [ ] Validate output quality

### Step 5: Run Experiments
- [ ] Generate full dataset (1000+ trajectories)
- [ ] Run ablation studies
- [ ] Collect results
- [ ] Compare against baselines

## 🎯 Immediate Next Steps (Ordered)

1. **Set up dFactory environment** (30 min)
   ```bash
   git clone https://github.com/inclusionAI/dFactory.git --recursive
   cd dFactory/VeOmni
   uv sync --extra gpu
   source .venv/bin/activate
   ```

2. **Download LLaDA2.0 model** (10 min)
   ```bash
   python scripts/download_hf_model.py \
       --repo_id inclusionAI/LLaDA2.0-mini-preview \
       --local_dir models/llada2_mini
   ```

3. **Study LLaDA2.0 API** (1-2 hours)
   - Read dFactory documentation
   - Examine `tasks/train_llada2_bd.py`
   - Understand model input/output format
   - Check tokenization approach

4. **Create Tokenization Layer** (2-3 hours)
   - Implement `src/models/llada_tokenizer.py`
   - Map MiniGrid obs → tokens
   - Map actions → tokens
   - Handle special tokens (MASK, etc.)

5. **Integrate with dFactory Training** (3-4 hours)
   - Create `tasks/train_any_order_babyai.py`
   - Adapt our training loop to dFactory's framework
   - Update configs
   - Test with small run

6. **Test and Debug** (2-3 hours)
   - Run integration tests
   - Fix any issues
   - Verify outputs

7. **Run First Real Experiment** (variable)
   - Generate data
   - Train for real
   - Evaluate
   - Analyze results

## 📊 Expected Timeline

- **Setup & Integration**: 1-2 days
- **Testing & Debugging**: 1 day
- **Small-scale experiments**: 2-3 days
- **Full ablation study**: 1 week
- **Analysis & reporting**: 2-3 days

**Total**: ~2 weeks for complete Phase 1

## 🔍 Key Files to Modify for Integration

1. `scripts/train.py` - Replace mock model with LLaDA2.0
2. `src/models/llada_wrapper.py` - NEW: Create LLaDA2.0 wrapper
3. `src/models/llada_tokenizer.py` - NEW: Create tokenization layer
4. `src/training/trainer.py` - Update forward pass
5. `configs/model_configs/llada2_mini.yaml` - NEW: Model config

## 💡 Tips for Integration

1. **Start Small**: Test with 1 batch before full training
2. **Log Everything**: Add extensive logging during integration
3. **Compare Outputs**: Verify shapes and values match expectations
4. **Use dFactory Examples**: Follow their patterns closely
5. **Test Incrementally**: Don't change everything at once

## 🆘 If You Get Stuck

1. Check dFactory's documentation
2. Look at similar tasks in dFactory/tasks/
3. Print tensor shapes liberally
4. Test components in isolation
5. Refer to LLaDA2.0 paper for architecture details

## ✨ What You Have Ready to Go

- Complete data pipeline
- All masking strategies implemented
- Training infrastructure
- Evaluation suite
- Configuration system
- Ablation experiment scripts
- Comprehensive documentation

**You can start testing the pipeline TODAY** with the mock model to verify everything works. Then swap in the real LLaDA2.0 model when ready!

## 📚 Additional Resources

- dFactory: https://github.com/inclusionAI/dFactory
- LLaDA2.0: https://huggingface.co/inclusionAI/LLaDA2.0-mini-preview
- MiniGrid: https://minigrid.farama.org/
- BabyAI: https://github.com/mila-iqia/babyai

## Contact & Support

This project is ready for:
- Pipeline testing (✅ now)
- LLaDA2.0 integration (🚧 next step)
- Experiment execution (📅 after integration)
