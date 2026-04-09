import pytest
import torch
from unittest.mock import MagicMock
from aomt.model.inference import generate_action

@pytest.fixture
def mock_components():
    model = MagicMock()
    tokenizer = MagicMock()
    tokenizer.mask_token_id = 156895
    tokenizer.eos_token_id = 156892
    
    # Mock apply_chat_template to return fixed IDs
    tokenizer.apply_chat_template.return_value = [1, 2, 3]
    tokenizer.encode.return_value = [1]

    
    # Mock model generate to return fixed IDs
    tokenizer.decode.return_value = "picked up potato"
    
    # Mock model generate call
    fake_output = torch.tensor([[1, 2, 3, 10, 20, 30]]) # prompt + generation
    model.generate.return_value = fake_output
    model.device = torch.device("cpu")
    
    return model, tokenizer

def test_generate_action_smoke(mock_components):
    model, tokenizer = mock_components
    history = ["You are in a kitchen.", "Observation: You see a potato."]
    
    res = generate_action(model, tokenizer, history)
    
    assert isinstance(res, str)
    assert res == "picked up potato"
    
    # Verify calls
    tokenizer.apply_chat_template.assert_called_once()
    model.generate.assert_called_once()

def test_generate_action_empty_history(mock_components):
    model, tokenizer = mock_components
    history = []
    
    res = generate_action(model, tokenizer, history)
    assert res == "picked up potato"
