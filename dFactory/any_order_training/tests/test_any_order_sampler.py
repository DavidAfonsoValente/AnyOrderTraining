
import unittest
import random
from any_order_training.sampler import AnyOrderMaskSampler

class TestAnyOrderMaskSampler(unittest.TestCase):

    def setUp(self):
        self.trajectory = ["obs1", "act1", "obs2", "act2", "obs3", "act3"]
        self.mask_token = "[MASK]"

    def test_initialization(self):
        """Test sampler initialization with valid and invalid probabilities."""
        AnyOrderMaskSampler(0.5)
        with self.assertRaises(ValueError):
            AnyOrderMaskSampler(-0.1)
        with self.assertRaises(ValueError):
            AnyOrderMaskSampler(1.1)

    def test_no_masking(self):
        """Test with mask_prob = 0.0, no items should be masked."""
        sampler = AnyOrderMaskSampler(mask_prob=0.0, mask_token=self.mask_token)
        masked_traj, labels, mask_indices = sampler(self.trajectory)
        self.assertEqual(masked_traj, self.trajectory)
        self.assertTrue(all(l is None for l in labels))
        self.assertFalse(any(mask_indices))

    def test_full_masking(self):
        """Test with mask_prob = 1.0, all items should be masked."""
        sampler = AnyOrderMaskSampler(mask_prob=1.0, mask_token=self.mask_token)
        masked_traj, labels, mask_indices = sampler(self.trajectory)
        
        # With the guarantee that at least one item is unmasked, we can't test for full masking
        # in a deterministic way. Instead, we check that almost everything is masked.
        # Let's check if n-1 items are masked.
        num_masked = sum(mask_indices)
        self.assertGreaterEqual(num_masked, len(self.trajectory) -1)


    def test_partial_masking(self):
        """Test with a probability between 0 and 1."""
        random.seed(42)
        sampler = AnyOrderMaskSampler(mask_prob=0.5, mask_token=self.mask_token)
        masked_traj, labels, mask_indices = sampler(self.trajectory)

        self.assertEqual(len(masked_traj), len(self.trajectory))
        self.assertEqual(len(labels), len(self.trajectory))
        self.assertEqual(len(mask_indices), len(self.trajectory))

        num_masked = 0
        for i, item in enumerate(self.trajectory):
            if mask_indices[i]:
                num_masked += 1
                self.assertEqual(masked_traj[i], self.mask_token)
                self.assertEqual(labels[i], item)
            else:
                self.assertEqual(masked_traj[i], item)
                self.assertIsNone(labels[i])
        
        self.assertGreater(num_masked, 0)
        self.assertLess(num_masked, len(self.trajectory))

    def test_empty_trajectory(self):
        """Test with an empty trajectory."""
        sampler = AnyOrderMaskSampler(mask_prob=0.5, mask_token=self.mask_token)
        masked_traj, labels, mask_indices = sampler([])
        self.assertEqual(masked_traj, [])
        self.assertEqual(labels, [])
        self.assertEqual(mask_indices, [])

    def test_ensure_one_masked(self):
        """Force a case where random chance might not mask anything."""
        random.seed(0) # A seed that might not produce masking with low prob
        sampler = AnyOrderMaskSampler(mask_prob=0.01, mask_token=self.mask_token)
        masked_traj, labels, mask_indices = sampler(self.trajectory)
        self.assertTrue(any(mask_indices))

    def test_ensure_one_unmasked(self):
        """Force a case where random chance might mask everything."""
        random.seed(0) # A seed that might produce full masking with high prob
        sampler = AnyOrderMaskSampler(mask_prob=0.99, mask_token=self.mask_token)
        masked_traj, labels, mask_indices = sampler(self.trajectory)
        self.assertFalse(all(mask_indices))

if __name__ == '__main__':
    unittest.main()
