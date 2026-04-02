import pytest
import numpy as np
from aomt.data.tokenize_trajectory import tokenize_trajectory
from aomt.data.masking import (
    apply_unit_mask,
    sample_aomt_mixed_mask
)

def test_apply_unit_mask_invariants(synthetic_trajectory, tokenizer_fixture):
    traj = tokenize_trajectory(synthetic_trajectory, tokenizer_fixture)
    mask_token_id = tokenizer_fixture.mask_token_id
    
    # Mask unit 1 (Action 0)
    masked_ids, labels = apply_unit_mask(traj.token_ids, traj.unit_spans, [1], mask_token_id)
    
    span = traj.unit_spans[1]
    # Check masked tokens
    assert all(tid == mask_token_id for tid in masked_ids[span.start:span.end])
    # Check labels
    assert labels[span.start:span.end] == traj.token_ids[span.start:span.end]
    # Check unmasked parts
    assert all(lid == -100 for lid in labels[:span.start])
    assert all(lid == -100 for lid in labels[span.end:])

def test_aomt_mixed_mask_bounds(synthetic_trajectory, tokenizer_fixture):
    traj = tokenize_trajectory(synthetic_trajectory, tokenizer_fixture)
    rng = np.random.default_rng(42)
    
    for _ in range(100):
        masked = sample_aomt_mixed_mask(traj.unit_spans, 0.5, rng)
        assert len(masked) >= 1
        assert len(masked) < len(traj.unit_spans)
