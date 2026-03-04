# tests/test_loss_computation.py
import unittest
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import numpy as np

from aomt.training.objectives import masked_unit_cross_entropy
from aomt.training.trainer import training_step

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

    def test_attention_mask_correctness(self):
        """
        Sanity Check 11.4: Verifies that AOMT (bidirectional) has a lower loss
        than standard SFT (causal) on the same data, proving that the more
        informative bidirectional attention mask is being used correctly.
        """
        print("\nRunning test: test_attention_mask_correctness")
        
        # Use a tiny, randomly initialized model for this test
        config = AutoConfig.from_pretrained("gpt2", n_layer=2, n_head=2)
        model = AutoModelForCausalLM.from_config(config).to("cpu")
        
        # A batch where an action is masked. SFT should only see the prefix,
        # while AOMT should see the whole trajectory.
        # [OBS_0] [ACT_0] [OBS_1]
        # Target is ACT_0
        input_ids = torch.randint(0, self.vocab_size, (1, 30))
        target_ids = input_ids.clone()
        loss_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        loss_mask[0, 10:20] = True # Mask the middle unit (ACT_0)
        
        attention_mask = torch.ones_like(input_ids)

        # --- SFT (Causal) Run ---
        sft_batch = {
            "input_ids": input_ids.clone(), # Don't replace with [MASK] for SFT
            "target_ids": target_ids.clone(),
            "loss_mask": loss_mask.clone(),
            "attention_mask": attention_mask.clone(),
            "use_causal_mask": torch.tensor([True])
        }
        sft_loss = training_step(model, sft_batch, device="cpu")
        
        # --- AOMT (Bidirectional) Run ---
        aomt_input_ids = input_ids.clone()
        aomt_input_ids[loss_mask] = 50257 # Use a fake mask token ID
        aomt_batch = {
            "input_ids": aomt_input_ids,
            "target_ids": target_ids.clone(),
            "loss_mask": loss_mask.clone(),
            "attention_mask": attention_mask.clone(),
            "use_causal_mask": torch.tensor([False])
        }
        aomt_loss = training_step(model, aomt_batch, device="cpu")
        
        print(f"Standard SFT (Causal) Loss: {sft_loss.item():.4f}")
        print(f"AOMT (Bidirectional) Loss: {aomt_loss.item():.4f}")
        
        # The core hypothesis: reconstructing with bidirectional context is easier,
        # hence the loss should be lower.
        self.assertLess(aomt_loss.item(), sft_loss.item())
        print("Test passed: AOMT loss is lower than SFT loss, as expected.")


if __name__ == '__main__':
    unittest.main()
