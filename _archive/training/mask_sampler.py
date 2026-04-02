"""
training/mask_sampler.py
Unit-level mask samplers for AOMT.
"""

from enum import Enum
import numpy as np

class MaskingMode(Enum):
    MIXED = "mixed"

class UnitMaskSampler:
    def __init__(self, mask_prob: float = 0.25):
        self.mask_prob = mask_prob

    def sample_mask(self, spans: list, mode: str, rng: np.random.Generator) -> list:
        """
        spans: list of (start, end, type)
        returns: list of indices into spans that should be masked.
        """
        if mode != "mixed":
            raise ValueError(f"Unsupported masking mode: {mode}")

        # Bernoulli(p) per unit
        masked_indices = []
        for i, (s, e, utype) in enumerate(spans):
            if rng.random() < self.mask_prob:
                masked_indices.append(i)

        # Force at least one masked unit
        if not masked_indices and len(spans) > 0:
            masked_indices.append(int(rng.integers(len(spans))))

        return masked_indices
