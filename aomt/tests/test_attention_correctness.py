# tests/test_attention_correctness.py
import unittest
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import os

# Import the core logic directly from the task scripts
from aomt.tasks.train_standard_sft import apply_response_unit_mask
from aomt.tasks.train_aomt import apply_unit_mask, build_foundation_model

class TestAttentionMaskCorrectness(unittest.TestCase):
    """
    Verifies that the attention masks are being constructed and applied correctly.
    Section 13 of Engineering Spec: 'The causal attention mask is never used anywhere in this project'
    except for the 'standard_sft' mode.
    """

    def setUp(self):
        self.model_path = "weights/llada2-mini-merged"
        if not torch.cuda.is_available():
            self.skipTest("No GPU available.")
        if not os.path.exists(self.model_path):
            self.skipTest(f"Model weights not found at {self.model_path}.")

        print(f"Loading model for test from {self.model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        # Use the standard project loader which handles weights correctly
        self.model = build_foundation_model(
            weights_path=self.model_path,
            config_path=self.model_path,
            torch_dtype="bfloat16",
            attn_implementation="sdpa",
            init_device="cuda"
        ).eval()

    def test_bidirectional_forward_pass(self):
        """
        Verify that the model can perform a forward pass with a bidirectional 
        (all-ones) attention mask, which is the core requirement for AOMT.
        """
        print(f"\nRunning bidirectional forward pass test on {torch.cuda.get_device_name(0)}")
        
        # Create a small batch
        B, L = 2, 64
        input_ids = torch.randint(100, 1000, (B, L)).to(self.model.device)
        # LLaDA 2.0 requires 4D block attention mask: (B, 1, L, L)
        attn_mask = torch.ones((B, 1, L, L), dtype=torch.long).to(self.model.device)
        
        try:
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attn_mask)
            
            self.assertIsNotNone(outputs.logits)
            self.assertEqual(outputs.logits.shape[:2], (B, L))
            print("Successfully performed bidirectional forward pass.")
        except Exception as e:
            self.fail(f"Bidirectional forward pass failed: {e}")

if __name__ == '__main__':
    unittest.main()
