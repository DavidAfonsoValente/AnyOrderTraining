import pytest
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from aomt.model.llada_wrapper import load_model_and_tokenizer
from aomt.model.inference import (
    generate_action,
    corrupt_observation
)
from aomt.data.dataset import AOMTDataset
from aomt.data.collator import AOMTDataCollator
from aomt.training.losses import masked_cross_entropy_loss
from aomt.evaluation.eval_alfworld import evaluate_alfworld
from omegaconf import OmegaConf

@pytest.fixture(scope="module")
def real_components():
    model_id = "aomt/weights/LLaDA2.0-mini"
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
    seq_len = input_ids.shape[1]
    attention_mask = torch.ones((1, 1, seq_len, seq_len)).to(model.device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    assert outputs.logits.shape[0] == 1

@pytest.mark.gpu
def test_generate_action_real(real_components):
    model, tokenizer = real_components
    history = ["You are in a kitchen.", "Observation: You see a potato."]
    # Use higher temp for smoke test to ensure the model bits flip
    res = generate_action(model, tokenizer, history, temperature=1.0, steps=8)
    assert isinstance(res, str)
    print(f"Smoke test generation: '{res}'")

@pytest.mark.gpu
def test_aomt_dataset_getitem_no_partial_masking_real(real_components):
    _, tokenizer = real_components
    raw_data = [{
        "conversations": [
            {"from": "human", "value": "Obs 0"},
            {"from": "gpt", "value": "Act 0"},
            {"from": "human", "value": "Obs 1"}
        ]
    }]
    ds = AOMTDataset(raw_data, tokenizer, method="aomt_mixed", p_mask=0.5)
    item = ds[0]
    input_ids = item["input_ids"]
    starts = item["unit_span_starts"]
    ends = item["unit_span_ends"]
    for start, end in zip(starts, ends):
        span = input_ids[start:end]
        is_masked = (span == tokenizer.mask_token_id).all().item()
        is_original = (span != tokenizer.mask_token_id).all().item()
        assert is_masked or is_original
