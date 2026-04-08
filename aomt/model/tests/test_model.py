import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock
from aomt.model.inference import generate_action

class MockModel(nn.Module):
    def __init__(self, vocab_size=1000):
        super().__init__()
        self.config = MagicMock()
        self.device = torch.device("cpu")
        self.vocab_size = vocab_size
        
    def forward(self, input_ids, **kwargs):
        # Return random logits (batch, seq, vocab)
        batch, seq = input_ids.shape
        logits = torch.randn(batch, seq, self.vocab_size)
        mock_output = MagicMock()
        mock_output.logits = logits
        return mock_output

    def to(self, *args, **kwargs):
        return self

@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.mask_token_id = 156895
    tokenizer.pad_token_id = 156892
    tokenizer.vocab_size = 156891
    tokenizer.mask_token = "<|mask|>"
    
    # Simple encoding: each word is a token ID
    def mock_encode(text, add_special_tokens=False, **kwargs):
        return [ord(c) for c in text[:10]] # Deterministic fake IDs
    
    tokenizer.encode.side_effect = mock_encode
    tokenizer.decode.side_effect = lambda ids, **kwargs: "".join([chr(int(i)) for i in ids])
    return tokenizer

