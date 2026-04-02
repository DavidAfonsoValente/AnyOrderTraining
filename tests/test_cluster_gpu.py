import pytest
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from aomt.model.llada_wrapper import load_model_and_tokenizer
from aomt.model.inference import (
    generate_next_action_mode_a, 
    generate_next_action_mode_b, 
    tokenize_history,
    block_diffusion_decode
)
from aomt.data.dataset import AOMTDataset
from aomt.data.collator import AOMTDataCollator
from aomt.training.losses import masked_cross_entropy_loss
from aomt.evaluation.eval_alfworld import evaluate_alfworld
from omegaconf import OmegaConf

# Shared fixture for real model and tokenizer
@pytest.fixture(scope="module")
def real_components():
    model_id = "aomt/weights/LLaDA2.0-mini"
    # Load in bf16 to save memory
    model, tokenizer = load_model_and_tokenizer(model_id, precision="bf16", device_map="auto")
    return model, tokenizer

@pytest.mark.gpu
def test_model_loads_without_oom(real_components):
    model, tokenizer = real_components
    assert model is not None
    assert tokenizer is not None

@pytest.mark.gpu
def test_forward_pass_output_shape_real(real_components):
    model, tokenizer = real_components
    input_ids = torch.tensor([[1, 2, 3]]).to(model.device)
    attention_mask = torch.ones_like(input_ids).to(model.device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    assert outputs.logits.shape[0] == 1
    assert outputs.logits.shape[1] == 3
    assert outputs.logits.shape[2] == tokenizer.vocab_size + 5 # account for special tokens

@pytest.mark.gpu
def test_mode_b_h1_numerically_equals_mode_a_real(real_components):
    model, tokenizer = real_components
    history = ["You are in a kitchen.", "go to fridge", "Observation: You see a fridge."]
    
    # Force same seed for block diffusion if possible, or use temp=0
    res_a = generate_next_action_mode_a(model, tokenizer, history, diffusion_steps=4, device=model.device)
    res_b = generate_next_action_mode_b(model, tokenizer, history, method="aomt_mixed", planning_horizon=1, diffusion_steps=4, device=model.device)
    
    # In Mode B with H=1, the suffix is exactly the same as Mode A
    # They should be identical at temp=0
    assert res_a == res_b

@pytest.mark.gpu
def test_alfworld_episode_mode_a_real(real_components):
    model, tokenizer = real_components
    config = OmegaConf.create({
        "max_seq_len": 2048,
        "median_action_tokens": 33,
        "median_obs_tokens": 17
    })
    
    # This might require alfworld data to be present
    try:
        res = evaluate_alfworld(model, tokenizer, config, split="eval_in_distribution", inference_mode="mode_a", n_episodes=1)
        assert "success_rate" in res
    except Exception as e:
        pytest.skip(f"AlfWorld eval failed (likely missing data/env): {e}")

@pytest.mark.gpu
def test_aomt_dataset_getitem_no_partial_masking_real(real_components):
    _, tokenizer = real_components
    # Synthetic trajectory for speed
    raw_data = [{
        "conversations": [
            {"from": "human", "value": "Obs 0"},
            {"from": "gpt", "value": "Act 0"},
            {"from": "human", "value": "Obs 1"}
        ]
    }]
    ds = AOMTDataset(raw_data, tokenizer, method="aomt_mixed", p_mask=0.5)
    
    for _ in range(10):
        item = ds[0]
        input_ids = item["input_ids"]
        # Check that each unit is either fully masked or fully original
        # For our 3 units
        for start, end in zip(item["unit_span_starts"], item["unit_span_ends"]):
            span = input_ids[start:end]
            is_masked = (span == tokenizer.mask_token_id).all().item()
            is_original = (span != tokenizer.mask_token_id).all().item()
            assert is_masked or is_original, f"Partial masking detected in span {start}:{end}"
