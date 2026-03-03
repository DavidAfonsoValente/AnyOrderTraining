import random
from typing import List, Optional, Tuple

class AnyOrderSampler:
    """
    Samples a mask for trajectory units based on a specified strategy.
    Corresponds to the sampler described in the project specification.
    """

    def __init__(
        self,
        mode: str,
        mask_prob: float = 0.5,
        mask_strategy: str = 'mixed', # 'mixed', 'observations_only', 'actions_only'
    ):
        """
        Initializes the sampler.

        Args:
            mode: The masking mode ('b_singleunit', 'b_prefix', 'any_order').
            mask_prob: The Bernoulli probability for 'any_order' mode.
            mask_strategy: Determines which types of units are candidates for masking.
        """
        if mode not in ['b_singleunit', 'b_prefix', 'any_order']:
            raise ValueError(f"Invalid sampler mode: {mode}")
        if mask_strategy not in ['mixed', 'observations_only', 'actions_only']:
            raise ValueError(f"Invalid mask strategy: {mask_strategy}")

        self.mode = mode
        self.mask_prob = mask_prob
        self.mask_strategy = mask_strategy
        print(f"Initialized AnyOrderSampler with mode='{self.mode}', strategy='{self.mask_strategy}', p={self.mask_prob}")

    def sample_mask(self, unit_boundaries: List[Tuple[int, int]], unit_types: List[str]) -> List[int]:
        """
        Generates a list of unit indices to be masked.

        Args:
            unit_boundaries: A list of (start, end) tuples for each unit.
            unit_types: A list of 'observation' or 'action' for each unit.

        Returns:
            A list of integer indices of the units to mask.
        """
        num_units = len(unit_boundaries)
        all_unit_indices = list(range(num_units))
        
        # 1. Determine the pool of candidate indices based on the strategy
        if self.mask_strategy == 'observations_only':
            candidate_indices = [i for i, t in enumerate(unit_types) if t == 'observation']
        elif self.mask_strategy == 'actions_only':
            candidate_indices = [i for i, t in enumerate(unit_types) if t == 'action']
        else: # 'mixed'
            candidate_indices = all_unit_indices

        if not candidate_indices:
            return []

        # 2. Apply the masking mode to the candidate indices
        mask = self._apply_mode(candidate_indices, all_unit_indices)

        # 3. Ensure mask is not empty (as per spec), if there are candidates
        if not mask and candidate_indices:
            return [random.choice(candidate_indices)]
        
        return sorted(list(set(mask))) # Return sorted unique indices

    def _apply_mode(self, candidate_indices: List[int], all_unit_indices: List[int]) -> List[int]:
        """Helper function to apply the specific masking logic."""
        
        if self.mode == 'b_singleunit':
            # Spec: "mask one action unit per step". Our candidates are already actions.
            return [random.choice(candidate_indices)]

        elif self.mode == 'b_prefix':
            # Spec: "mask all units from split k onward".
            if len(all_unit_indices) < 2:
                return [] # Cannot create a prefix mask on a single-unit trajectory
            k = random.randint(1, len(all_unit_indices) - 1)
            return all_unit_indices[k:]

        elif self.mode == 'any_order':
            # Spec: "Bernoulli(mask_prob) independent unit masking" on candidates.
            return [i for i in candidate_indices if random.random() < self.mask_prob]

        return [] # Should not be reached
