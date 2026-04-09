import pytest
from aomt.data.tests.conftest import tokenizer_fixture, synthetic_trajectory

def pytest_configure(config):
    # Register the 'gpu' marker to avoid warnings
    config.addinivalue_line("markers", "gpu: marks tests as requiring a GPU to run")
