import pytest
import torch
from transformers import AutoTokenizer

@pytest.fixture(scope="session")
def tokenizer_fixture():
    # Use the local weights path
    return AutoTokenizer.from_pretrained("aomt/weights/LLaDA2.0-mini")

@pytest.fixture
def synthetic_trajectory():
    return {
        "conversations": [
            {"from": "human", "value": "Observation 0"},
            {"from": "gpt", "value": "Action 0"},
            {"from": "human", "value": "Observation 1"},
            {"from": "gpt", "value": "Action 1"},
            {"from": "human", "value": "Observation 2"}
        ]
    }
