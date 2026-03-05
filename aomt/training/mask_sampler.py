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
    This is the core logic for the different training objectives.

    Args:
        units (List[TokenizedUnit]): The list of all unit spans in a trajectory.
        mask_prob (float): The probability of masking a unit (for stochastic modes).
        mode (MaskMode): The masking strategy to apply.
        rng (np.random.Generator): A random number generator for reproducibility.

    Returns:
        List[TokenizedUnit]: A list of the specific units that should be masked.
    """
    if mode == MaskMode.STANDARD_SFT:
        # For diffusion-based SFT, we mask the final action and predict it.
        # Find the last unit with type "act" in the trajectory.
        last_action_unit = None
        for u in reversed(units):
            if u.unit_type == "act":
                last_action_unit = u
                break
        return [last_action_unit] if last_action_unit else []

    elif mode == MaskMode.PREFIX_SFT_STAGE1:
        # This mode is handled by a special collator (`build_prefix_sft_examples`)
        # which creates many small examples from a single trajectory.
        # This function should not be called directly for Stage 1.
        raise NotImplementedError(
            "PREFIX_SFT_STAGE1 masking is handled by the collator, not the sampler."
        )

    elif mode == MaskMode.PREFIX_SFT_STAGE2:
        # This mode is also handled by a special collator.
        raise NotImplementedError(
            "PREFIX_SFT_STAGE2 masking is handled by the collator, not the sampler."
        )

    elif mode == MaskMode.ACTION_ONLY:
        # AOMT-Action-Only: Randomly mask a subset of *action* units.
        return [u for u in units if u.unit_type == "act" and rng.random() < mask_prob]

    elif mode == MaskMode.MIXED:
        # AOMT-Mixed: Randomly mask thought, action, and observation units, but NOT the objective.
        units_to_mask = []
        for i, u in enumerate(units):
            # The objective is the first unit (index 0) and has type "obs".
            # We explicitly exclude the objective from masking.
            is_objective = i == 0 and u.unit_type == "obs"
            
            # Thoughts and actions are contained within "act" units.
            is_action_or_thought = u.unit_type == "act"

            # Observations are "obs" units that are not the objective.
            is_observation = u.unit_type == "obs" and not is_objective

            if (is_action_or_thought or is_observation) and rng.random() < mask_prob:
                units_to_mask.append(u)
        return units_to_mask

    else:
        raise ValueError(f"Unknown MaskMode: {mode}")

def apply_unit_mask(
    tokenized_traj: TokenizedTrajectory,
    mask_prob: float,
    mode: MaskMode,
    mask_token_id: int,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """
    Applies masking to a tokenized trajectory based on the selected mode.

    It returns a new tensor with mask tokens and a boolean mask indicating
    which tokens contribute to the loss.

    Args:
        tokenized_traj (TokenizedTrajectory): The input trajectory to mask.
        mask_prob (float): Probability of masking for stochastic modes.
        mode (MaskMode): The masking strategy.
        mask_token_id (int): The ID of the `[MASK]` token.
        rng (Optional[np.random.Generator]): Random number generator. If None, a new
                                             one is created.

    Returns:
        Tuple[torch.LongTensor, torch.LongTensor]:
        - masked_input_ids: A copy of the input IDs with selected units replaced
                            by the `mask_token_id`.
        - loss_mask: A boolean tensor of the same shape, where `True` marks
                     the positions of the masked units for loss calculation.
    """
    if rng is None:
        rng = np.random.default_rng()

    input_ids = tokenized_traj.input_ids.clone()
    loss_mask = torch.zeros_like(input_ids, dtype=torch.bool)

    # Decide which units to mask based on the strategy
    units_to_mask = _select_units_to_mask(
        tokenized_traj.unit_spans, mask_prob, mode, rng
    )

    # Apply the mask to the selected units
    for unit in units_to_mask:
        if unit.token_start < unit.token_end: # Ensure span is valid
            # For all diffusion-based modes, we replace the target tokens
            # with the mask token.
            input_ids[unit.token_start:unit.token_end] = mask_token_id
            
            # The loss mask is True for all tokens within the targeted unit.
            loss_mask[unit.token_start:unit.token_end] = True

    return input_ids, loss_mask
