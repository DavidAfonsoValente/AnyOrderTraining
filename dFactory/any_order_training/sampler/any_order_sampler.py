
import random
from typing import List, Tuple, Any

class AnyOrderMaskSampler:
    """
    Samples a mask for a trajectory of observations and actions.
    """

    def __init__(self, mask_prob: float, mask_token: str = "[MASK]"):
        """
        Initializes the sampler.

        Args:
            mask_prob: The probability of masking an item in the trajectory.
            mask_token: The token to use for masking.
        """
        if not 0.0 <= mask_prob <= 1.0:
            raise ValueError("Mask probability must be between 0 and 1.")
        self.mask_prob = mask_prob
        self.mask_token = mask_token

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

        for item in trajectory:
            if random.random() < self.mask_prob:
                masked_trajectory.append(self.mask_token)
                labels.append(item)
                mask_indices.append(True)
            else:
                masked_trajectory.append(item)
                labels.append(None) # Using None as a placeholder for unmasked items
                mask_indices.append(False)

        # Ensure at least one item is masked if the trajectory is not empty
        if self.mask_prob > 0 and not any(mask_indices) and trajectory:
            idx_to_mask = random.randint(0, len(trajectory) - 1)
            masked_trajectory[idx_to_mask] = self.mask_token
            labels[idx_to_mask] = trajectory[idx_to_mask]
            mask_indices[idx_to_mask] = True

        # Ensure at least one item is not masked if the trajectory is not empty
        if self.mask_prob < 1 and all(mask_indices) and trajectory:
            idx_to_unmask = random.randint(0, len(trajectory) - 1)
            masked_trajectory[idx_to_unmask] = trajectory[idx_to_unmask]
            labels[idx_to_unmask] = None
            mask_indices[idx_to_unmask] = False

        return masked_trajectory, labels, mask_indices

