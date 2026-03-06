# tests/test_loss_computation.py
import unittest
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import numpy as np

from ..training.objectives import masked_unit_cross_entropy
from ..training.trainer import training_step

class TestLossComputation(unittest.TestCase):

    def setUp(self):
        """Set up mock logits, targets, and masks for testing."""
        self.vocab_size = 100
        self.seq_len = 20
        self.batch_size = 2

        # Create realistic-looking data
        self.logits = torch.randn(self.batch_size, self.seq_len, self.vocab_size)
        self.target_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        
        # Create a loss mask that targets specific regions
        self.loss_mask = torch.zeros(self.batch_size, self.seq_len, dtype=torch.bool)
        self.loss_mask[0, 5:10] = True  # Batch 0, tokens 5-9
        self.loss_mask[1, 15:20] = True # Batch 1, tokens 15-19

    def test_masked_cross_entropy_calculation(self):
        """
        Sanity Check 11.3: Verifies that the loss function correctly computes
        the mean cross-entropy over only the masked positions.
        """
        print("\nRunning test: test_masked_cross_entropy_calculation")
        # --- 1. Calculate loss using the function under test ---
        computed_loss = masked_unit_cross_entropy(self.logits, self.target_ids, self.loss_mask)

        # --- 2. Calculate loss manually for verification ---
        total_loss = 0.0
        masked_token_count = 0
        
        for i in range(self.batch_size):
            for j in range(self.seq_len):
                if self.loss_mask[i, j]:
                    # F.cross_entropy expects (N, C) and (N)
                    token_loss = F.cross_entropy(
                        self.logits[i, j].unsqueeze(0),
                        self.target_ids[i, j].unsqueeze(0)
                    )
                    total_loss += token_loss
                    masked_token_count += 1
        
        manual_loss = total_loss / masked_token_count
        
        self.assertAlmostEqual(computed_loss.item(), manual_loss.item(), places=5)
        print("Test passed.")

    def test_zero_mask_returns_zero_loss(self):
        """
        Sanity Check 11.3: Verifies that if no tokens are masked, the loss is 0.
        """
        print("\nRunning test: test_zero_mask_returns_zero_loss")
        zero_mask = torch.zeros(self.batch_size, self.seq_len, dtype=torch.bool)
        loss = masked_unit_cross_entropy(self.logits, self.target_ids, zero_mask)
        self.assertEqual(loss.item(), 0.0)
        print("Test passed.")
