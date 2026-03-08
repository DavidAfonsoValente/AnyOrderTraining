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

    def test_gradients_only_from_masked_positions(self):
        """
        Sanity Check 11.3: Verifies that gradients are non-zero only for tokens
        that contributed to the loss (i.e., where loss_mask is True).
        """
        print("\nRunning test: test_gradients_only_from_masked_positions")
        
        # --- 1. Setup a minimal model and data ---
        # Make IDs simple and non-overlapping to easily check gradients
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long) # Batch size 1
        target_ids = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
        loss_mask = torch.tensor([[False, False, True, True, False, False]], dtype=torch.bool)
        
        # A simple model: just an embedding layer
        embedding = torch.nn.Embedding(self.vocab_size, 16)
        # The embedding weights require gradients
        embedding.weight.requires_grad = True

        # --- 2. Forward pass and loss calculation ---
        embedded_inputs = embedding(input_ids)
        # Create dummy logits; we only care about the gradient flow
        # We need to make this operation differentiable wrt embeddings
        dummy_logits = torch.randn(1, embedded_inputs.shape[1], self.vocab_size, requires_grad=True)

        # To make the connection, we can add the embedded inputs to a slice of logits
        # This is a bit of a hack, but it establishes the graph for autograd
        # The actual values don't matter, only the gradient path.
        dummy_logits[:, :, :16] += embedded_inputs

        loss = masked_unit_cross_entropy(dummy_logits, target_ids, loss_mask)
        
        # --- 3. Backward pass ---
        loss.backward()

        # --- 4. Check gradients on the embedding weights ---
        grads = embedding.weight.grad
        self.assertIsNotNone(grads, "Gradients were not computed.")

        # The IDs of the context tokens (loss_mask is False)
        context_token_ids = [1, 2, 5, 6]
        # The IDs of the target tokens (loss_mask is True)
        target_token_ids = [3, 4]

        # Gradients for context tokens should be zero
        for token_id in context_token_ids:
            self.assertEqual(
                grads[token_id].sum().item(), 0,
                f"Gradient for context token {token_id} should be zero but was not."
            )

        # Gradients for target tokens should be non-zero
        for token_id in target_token_ids:
            self.assertNotEqual(
                grads[token_id].sum().item(), 0,
                f"Gradient for target token {token_id} should be non-zero but was zero."
            )
        print("Test passed.")
