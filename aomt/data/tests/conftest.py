import pytest
from transformers import AutoTokenizer
import os
import shutil

@pytest.fixture(scope="session")
def tokenizer_fixture():
    # Force trust_remote_code to avoid interactive prompts in tests
    return AutoTokenizer.from_pretrained("aomt/weights/LLaDA2.0-mini", trust_remote_code=True)

@pytest.fixture(scope="session")
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

@pytest.fixture(autouse=True)
def clean_test_cache():
    # Ensure tests don't load stale cluster data
    cache_dir = "data/cache_test"
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    yield
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
