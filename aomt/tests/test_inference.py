import pytest
import torch
from unittest.mock import MagicMock
from aomt.model.inference import generate_next_action_mode_a, generate_next_action_mode_b

@pytest.fixture
def mock_components():
    model = MagicMock()
    tokenizer = MagicMock()
    tokenizer.mask_token_id = 156895
    tokenizer.encode.return_value = [1, 2, 3] # Fixed length
    tokenizer.decode.return_value = "action"
    
    # Mock model output
    mock_output = MagicMock()
    mock_output.logits = torch.randn(1, 500, 1000)
    model.return_value = mock_output
    model.to.return_value = model
    
    return model, tokenizer

def test_mode_b_h1_matches_mode_a(mock_components):
    model, tokenizer = mock_components
    history = ["O0", "A0", "O1"]
    
    import aomt.model.inference as inference
    original_decode = inference.block_diffusion_decode
    
    # Correct signature for block_diffusion_decode (6 arguments including defaults)
    def mock_decode(model, input_ids, prompt_len, mask_token_id, diffusion_steps=32, temperature=0.0):
        # Simply return the input_ids as if they were denoised to a fixed value
        return torch.zeros_like(input_ids)
        
    inference.block_diffusion_decode = MagicMock(side_effect=mock_decode)
    
    try:
        # Mode A call
        res_a = generate_next_action_mode_a(model, tokenizer, history, device="cpu")
        # Mode B call with H=1
        res_b = generate_next_action_mode_b(model, tokenizer, history, method="aomt_mixed", planning_horizon=1, device="cpu")
        
        # Hard assertion: they must match
        assert res_a == res_b
    finally:
        inference.block_diffusion_decode = original_decode

def test_mode_b_raises_for_invalid_horizon(mock_components):
    model, tokenizer = mock_components
    with pytest.raises(ValueError, match="planning_horizon must be >= 1"):
        generate_next_action_mode_b(model, tokenizer, ["O0"], planning_horizon=0, device="cpu")

def test_mode_b_raises_for_standard_sft(mock_components):
    model, tokenizer = mock_components
    with pytest.raises(ValueError, match="Mode B is ONLY valid for aomt_mixed"):
        generate_next_action_mode_b(model, tokenizer, ["O0"], method="standard_sft", device="cpu")
