# aomt/training/mask_sampler.py
import torch
import numpy as np
from enum import Enum
from typing import Tuple, List, Optional

from aomt.data.unit_parser import TokenizedTrajectory, TokenizedUnit

class MaskMode(Enum):
    """
    Defines the different masking strategies for training.
    Corresponds to the baselines and proposed methods in the paper.
    """
    STANDARD_SFT = "standard_sft"
    PREFIX_SFT_STAGE1 = "prefix_sft_stage1"
    PREFIX_SFT_STAGE2 = "prefix_sft_stage2"
    ACTION_ONLY = "action_only"
    MIXED = "mixed"

def _select_units_to_mask(
    units: List[TokenizedUnit],
    mask_prob: float,
    mode: MaskMode,
    rng: np.random.Generator,
) -> List[TokenizedUnit]:
    """
    Selects a list of units to be masked based on the specified mode.
    """
    if mode == MaskMode.STANDARD_SFT:
        return [u for u in units if u.unit_type == "act"]

    elif mode == MaskMode.ACTION_ONLY:
        # AOMT-Action-Only: Randomly mask a subset of *action* units.
        return [u for u in units if u.unit_type == "act" and rng.random() < mask_prob]

    elif mode == MaskMode.MIXED:
        # AOMT-Mixed: Randomly mask thought, action, and observation units, but NOT the objective.
        units_to_mask = []
        for i, u in enumerate(units):
            # The objective is the first unit (index 0).
            is_objective = i == 0
            
            # Observations are "obs" units that are not the objective.
            is_observation = u.unit_type == "obs" and not is_objective
            is_action = u.unit_type == "act"

            if (is_action or is_observation) and rng.random() < mask_prob:
                units_to_mask.append(u)
        return units_to_mask

    else:
        return []

def apply_unit_mask(
    tokenized_traj: TokenizedTrajectory,
    mask_prob: float,
    mode: MaskMode,
    mask_token_id: int,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """
    Applies masking to a tokenized trajectory based on the selected mode.
    Uses LLaDA2/MDLM random masking ratio within each selected unit:
    samples t ~ U(0,1) per sample, then independently masks each token
    in the selected units with probability t.
    """
    if rng is None:
        rng = np.random.default_rng()

    input_ids = tokenized_traj.input_ids.clone()
    loss_mask = torch.zeros_like(input_ids, dtype=torch.bool)

    # Decide which units to mask based on the strategy
    units_to_mask = _select_units_to_mask(
        tokenized_traj.unit_spans, mask_prob, mode, rng
    )

    # Force-mask at least one unit for AOMT modes if none were selected stochastically
    if not units_to_mask and mode in [MaskMode.ACTION_ONLY, MaskMode.MIXED]:
        eligible = []
        for i, u in enumerate(tokenized_traj.unit_spans):
            if mode == MaskMode.ACTION_ONLY and u.unit_type == "act":
                eligible.append(u)
            elif mode == MaskMode.MIXED and i > 0: # Anything but the objective
                eligible.append(u)
        
        if eligible:
            units_to_mask = [rng.choice(eligible)]

    # Sample random masking ratio t ~ U(0,1) for MDLM forward process
    t = rng.random()

    # Apply per-token random masking within the selected units
    for unit in units_to_mask:
        if unit.token_start < unit.token_end:
            span_len = unit.token_end - unit.token_start
            # Each token independently masked with probability t
            token_mask = torch.from_numpy(rng.random(span_len) < t)
            # Ensure at least 1 token is masked per unit
            if not token_mask.any():
                token_mask[rng.integers(span_len)] = True
            input_ids[unit.token_start:unit.token_end][token_mask] = mask_token_id
            loss_mask[unit.token_start:unit.token_end][token_mask] = True

    return input_ids, loss_mask
