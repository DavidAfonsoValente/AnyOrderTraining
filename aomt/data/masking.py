from typing import List, Tuple, Optional
import numpy as np
from .tokenize_trajectory import UnitSpan

def apply_unit_mask(
    token_ids: List[int],
    unit_spans: List[UnitSpan],
    masked_unit_indices: List[int],
    mask_token_id: int,
) -> Tuple[List[int], List[int]]:
    """
    Applies unit-level masking and returns (masked_input_ids, labels).
    """
    masked_input_ids = list(token_ids)
    labels = [-100] * len(token_ids)
    
    for idx in masked_unit_indices:
        span = unit_spans[idx]
        labels[span.start:span.end] = token_ids[span.start:span.end]
        masked_input_ids[span.start:span.end] = [mask_token_id] * (span.end - span.start)
        
    return masked_input_ids, labels

def sample_sft_mask(
    unit_spans: List[UnitSpan],
    step_t: int,
) -> List[int]:
    """Returns [index of the action unit at step t] for Standard SFT."""
    for i, span in enumerate(unit_spans):
        if span.unit_type == "action" and span.step_idx == step_t:
            return [i]
    return []

def sample_prefix_stage1_mask(
    unit_spans: List[UnitSpan],
    step_t: int,
) -> List[int]:
    """Returns [index of the observation unit at step t+1] for Prefix SFT Stage 1."""
    for i, span in enumerate(unit_spans):
        if span.unit_type == "observation" and span.step_idx == step_t + 1:
            return [i]
    return []

def sample_aomt_mixed_mask(
    unit_spans: List[UnitSpan],
    p_mask: float,
    rng: np.random.Generator,
    token_level: bool = False,
    token_ids: Optional[List[int]] = None,
) -> List[int]:
    """
    Samples the AOMT-Mixed mask (Bernoulli per unit).
    """
    indices = list(range(len(unit_spans)))
    masked_indices = []
    
    if not token_level:
        for i in indices:
            if rng.random() < p_mask:
                masked_indices.append(i)
    else:
        # Ablation A2: token-level driven unit masking
        if token_ids is None:
            raise ValueError("token_ids must be provided for token_level masking")
        for i in indices:
            span = unit_spans[i]
            # If any token in unit is drawn
            n_tokens = span.end - span.start
            if any(rng.random() < p_mask for _ in range(n_tokens)):
                masked_indices.append(i)
            
    # Guarantee 1 <= len(masked) <= len(spans) - 1
    if not masked_indices:
        masked_indices = [int(rng.choice(indices))]
    elif len(masked_indices) == len(indices):
        to_unmask = int(rng.choice(indices))
        masked_indices.remove(to_unmask)
        
    return masked_indices
