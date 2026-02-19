import random
from typing import List, Tuple, Any

class AnyOrderMaskSampler:
    """
    Samples a mask for a trajectory of observations and actions.
    """

    def __init__(self, mask_prob: float, mask_token: str = "[MASK]", masking_strategy: str = "all"):
        """
        Initializes the sampler.

        Args:
            mask_prob: The probability of masking an item in the trajectory.
            mask_token: The token to use for masking.
            masking_strategy: One of "all", "observations", or "actions".
                              Determines which items are candidates for masking.
        """
        if not 0.0 <= mask_prob <= 1.0:
            raise ValueError("Mask probability must be between 0 and 1.")
        if masking_strategy not in ["all", "observations", "actions"]:
            raise ValueError("masking_strategy must be 'all', 'observations', or 'actions'.")
        
        self.mask_prob = mask_prob
        self.mask_token = mask_token
        self.masking_strategy = masking_strategy

    def __call__(self, trajectory: List[Any]) -> Tuple[List[Any], List[Any], List[bool]]:
        """
        Applies masking to a trajectory.

        Args:
            trajectory: A list of observations and actions.

        Returns:
            A tuple containing:
            - The masked trajectory.
            - A list of labels for the masked items.
            - A list of booleans indicating which items were masked.
        """
        masked_trajectory = []
        labels = []
        mask_indices = []
        
        candidate_indices = []
        for i in range(len(trajectory)):
            is_obs = (i % 2 == 0)
            if self.masking_strategy == "all":
                candidate_indices.append(i)
            elif self.masking_strategy == "observations" and is_obs:
                candidate_indices.append(i)
            elif self.masking_strategy == "actions" and not is_obs:
                candidate_indices.append(i)

        # Decide which candidates to mask
        num_to_mask = int(len(candidate_indices) * self.mask_prob)
        indices_to_mask = set(random.sample(candidate_indices, num_to_mask))

        for i, item in enumerate(trajectory):
            if i in indices_to_mask:
                masked_trajectory.append(self.mask_token)
                labels.append(item)
                mask_indices.append(True)
            else:
                masked_trajectory.append(item)
                labels.append(None)
                mask_indices.append(False)
        
        return masked_trajectory, labels, mask_indices
