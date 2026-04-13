import pytest
from aomt.data.dataset import AOMTDataset

def test_standard_sft_dataset(synthetic_trajectory, tokenizer_fixture):
    raw_dataset = [synthetic_trajectory]
    # Optimized behavior: 1 trajectory produces 1 example with all actions masked
    ds = AOMTDataset(raw_dataset, tokenizer_fixture, method="standard_sft")
    assert len(ds) == 1
    
    # Check first example
    item = ds[0]
    # Decoded input_ids should end with the mask tokens for Action 0
    decoded = tokenizer_fixture.decode(item["input_ids"])
    # Standard SFT: O0, A0_masked
    assert "Observation 0" in decoded
    assert tokenizer_fixture.mask_token in decoded

def test_prefix_sft_stage1_dataset(synthetic_trajectory, tokenizer_fixture):
    raw_dataset = [synthetic_trajectory]
    # T=2 actions, but for prefix_sft_stage1 (Ot, At, Ot+1_masked)
    # Step 0: O0, A0, O1_masked
    # Step 1: O1, A1, O2_masked
    ds = AOMTDataset(raw_dataset, tokenizer_fixture, method="prefix_sft_stage1")
    assert len(ds) == 2
    
    item = ds[0]
    decoded = tokenizer_fixture.decode(item["input_ids"])
    # Should contain O0 and A0, and mask for O1
    assert "Observation 0" in decoded
    assert "Action 0" in decoded
    assert tokenizer_fixture.mask_token in decoded
    # Should NOT contain O2 or A1
    assert "Action 1" not in decoded
    assert "Observation 2" not in decoded

def test_aomt_mixed_dataset(synthetic_trajectory, tokenizer_fixture):
    raw_dataset = [synthetic_trajectory]
    # AOMT-Mixed: 1 example per trajectory
    ds = AOMTDataset(raw_dataset, tokenizer_fixture, method="aomt_mixed")
    assert len(ds) == 1
    
    item = ds[0]
    assert "input_ids" in item
    assert "labels" in item
    # Ensure some tokens are masked and some are not
    masked_count = (item["input_ids"] == tokenizer_fixture.mask_token_id).sum().item()
    assert masked_count > 0
    # Ensure some context remains
    unmasked_count = (item["input_ids"] != tokenizer_fixture.mask_token_id).sum().item()
    assert unmasked_count > 0
