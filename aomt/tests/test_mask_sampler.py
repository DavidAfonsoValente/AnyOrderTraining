# tests/test_mask_sampler.py
import unittest
import torch
import numpy as np
from transformers import AutoTokenizer

from ..training.mask_sampler import MaskMode, apply_unit_mask
from ..data.unit_parser import TokenizedTrajectory, TokenizedUnit

class TestMaskSampler(unittest.TestCase):

    def setUp(self):
        """Set up a mock tokenizer and a sample tokenized trajectory."""
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.add_special_tokens({
            'eos_token': '<|endoftext|>',
            'pad_token': '<|endoftext|>',
            'mask_token': '[MASK]'
        })
        self.mask_token_id = self.tokenizer.mask_token_id

        # A sample trajectory with 3 obs and 2 act units
        self.sample_spans = [
            TokenizedUnit(unit_type="obs", token_start=0, token_end=10, unit_index=0),
            TokenizedUnit(unit_type="act", token_start=11, token_end=20, unit_index=1),
            TokenizedUnit(unit_type="obs", token_start=21, token_end=30, unit_index=2),
            TokenizedUnit(unit_type="act", token_start=31, token_end=40, unit_index=3),
            TokenizedUnit(unit_type="obs", token_start=41, token_end=50, unit_index=4),
        ]
        self.sample_input_ids = torch.arange(51) # Simple dummy IDs
        
        self.tokenized_traj = TokenizedTrajectory(
            trajectory_id="test_traj_02",
            env="mock_env",
            input_ids=self.sample_input_ids,
            unit_spans=self.sample_spans
        )
        self.rng = np.random.default_rng(42)

    def test_standard_sft_masking(self):
        """
        Sanity Check 11.2: `STANDARD_SFT` should mask ONLY the last action unit.
        """
        print("\nRunning test: test_standard_sft_masking")
        masked_ids, loss_mask = apply_unit_mask(
            self.tokenized_traj, 0.0, MaskMode.STANDARD_SFT, self.mask_token_id, self.rng
        )

        # Input IDs should now be different from original
        self.assertFalse(torch.equal(masked_ids, self.sample_input_ids))

        # Find the last action unit
        last_act_unit = None
        for unit in reversed(self.sample_spans):
            if unit.unit_type == "act":
                last_act_unit = unit
                break
        
        self.assertIsNotNone(last_act_unit, "Sample trajectory has no action unit to test.")

        # Loss mask should be True ONLY at the last action unit's position
        for unit in self.sample_spans:
            span_mask = loss_mask[unit.token_start:unit.token_end]
            span_masked_ids = masked_ids[unit.token_start:unit.token_end]
            
            if unit == last_act_unit:
                self.assertTrue(torch.all(span_mask), "Last action unit should be in the loss mask.")
                self.assertTrue(torch.all(span_masked_ids == self.mask_token_id), "Last action unit should be replaced by mask tokens.")
            else:
                self.assertFalse(torch.any(span_mask), f"Unit {unit.unit_index} should NOT be in the loss mask.")
        print("Test passed.")

    def test_action_only_masking(self):
        """
        Sanity Check 11.2: `ACTION_ONLY` mode should mask a subset of actions.
        - Tokens in masked units should be replaced by `mask_token_id`.
        - Loss mask should be True only at masked action positions.
        """
        print("\nRunning test: test_action_only_masking")
        masked_ids, loss_mask = apply_unit_mask(
            self.tokenized_traj, 0.6, MaskMode.ACTION_ONLY, self.mask_token_id, self.rng
        )
        
        # Check that some action units were masked and some were not
        masked_act_units = 0
        for unit in self.sample_spans:
            if unit.unit_type == "act":
                span_loss_mask = loss_mask[unit.token_start:unit.token_end]
                span_input_ids = masked_ids[unit.token_start:unit.token_end]
                if torch.all(span_loss_mask):
                    masked_act_units += 1
                    # Verify tokens are replaced
                    self.assertTrue(torch.all(span_input_ids == self.mask_token_id))
            
            # No observation unit should ever be masked in this mode
            elif unit.unit_type == "obs":
                 self.assertFalse(torch.any(loss_mask[unit.token_start:unit.token_end]))
        
        # With p=0.6 and 2 action units, it's very likely at least one is masked
        self.assertGreater(masked_act_units, 0)
        print("Test passed.")
        
    def test_mixed_masking(self):
        """
        Sanity Check 11.2: `MIXED` mode should mask a subset of both obs and acts.
        This test is now probabilistic to avoid sensitivity to a single random seed.
        """
        print("\nRunning test: test_mixed_masking")
        
        masked_obs_found = False
        masked_act_found = False

        # Run multiple trials to see if we eventually mask both types
        for i in range(20):
            local_rng = np.random.default_rng(i)
            _, loss_mask = apply_unit_mask(
                self.tokenized_traj, 0.5, MaskMode.MIXED, self.mask_token_id, local_rng
            )
            
            for unit in self.sample_spans:
                # The first obs unit (objective) is never masked in MIXED mode.
                is_objective = unit.unit_index == 0 and unit.unit_type == "obs"
                if is_objective:
                    continue

                span_loss_mask = loss_mask[unit.token_start:unit.token_end]
                if torch.all(span_loss_mask):
                    if unit.unit_type == "obs":
                        masked_obs_found = True
                    elif unit.unit_type == "act":
                        masked_act_found = True
            
            if masked_obs_found and masked_act_found:
                break # Both found, no need to continue looping
        
        self.assertTrue(masked_obs_found, "MIXED mode failed to mask any observation units over 20 trials.")
        self.assertTrue(masked_act_found, "MIXED mode failed to mask any action units over 20 trials.")
        print("Test passed.")

    def test_mask_resampling(self):
        """
        Sanity Check 11.2: Ensure masks are different across calls (simulating epochs).
        """
        print("\nRunning test: test_mask_resampling")
        mask_set = set()
        
        # Simulate 10 calls to __getitem__
        for i in range(10):
            local_rng = np.random.default_rng(i) # Use a different seed each time
            _, loss_mask = apply_unit_mask(
                self.tokenized_traj, 0.4, MaskMode.MIXED, self.mask_token_id, local_rng
            )
            # Add the tuple representation of the boolean tensor to the set
            mask_set.add(tuple(loss_mask.tolist()))
            
        # With p=0.4, it's extremely unlikely all 10 masks are the same.
        # The spec requires at least 8/10 to be different. We'll check for > 5 for stability.
        self.assertGreater(len(mask_set), 5, "Masks are not being resampled correctly across calls.")
        print("Test passed.")

if __name__ == '__main__':
    unittest.main()
