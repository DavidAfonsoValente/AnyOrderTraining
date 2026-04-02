import pytest
from aomt.data.tokenize_trajectory import tokenize_trajectory

def test_tokenize_trajectory_count(synthetic_trajectory, tokenizer_fixture):
    traj = tokenize_trajectory(synthetic_trajectory, tokenizer_fixture)
    assert traj is not None
    # 2 actions
    assert traj.trajectory_length == 2
    # 5 units
    assert len(traj.unit_spans) == 5

def test_unit_spans_alternating(synthetic_trajectory, tokenizer_fixture):
    traj = tokenize_trajectory(synthetic_trajectory, tokenizer_fixture)
    types = [s.unit_type for s in traj.unit_spans]
    assert types == ["observation", "action", "observation", "action", "observation"]

def test_unit_spans_step_idx(synthetic_trajectory, tokenizer_fixture):
    traj = tokenize_trajectory(synthetic_trajectory, tokenizer_fixture)
    steps = [s.step_idx for s in traj.unit_spans]
    # O0(0), A0(0), O1(1), A1(1), O2(2)
    assert steps == [0, 0, 1, 1, 2]
