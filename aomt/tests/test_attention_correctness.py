# tests/test_attention_correctness.py
import unittest
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

# Import the core masking logic directly from the task scripts
from aomt.tasks.train_standard_sft import apply_response_unit_mask
from aomt.tasks.train_aomt import apply_unit_mask

class TestAttentionMaskCorrectness(unittest.TestCase):
    """
    Verifies that the attention masks are being constructed and applied correctly.
    Section 13 of Engineering Spec: 'The causal attention mask is never used anywhere in this project'
    except for the 'standard_sft' mode.
    """

    @classmethod
    def setUpClass(cls):
        # Use class method to load model once for all tests in this class
        cls.model_path = "models/LLaDA2.0-mini"
        if not torch.cuda.is_available() or not torch.os.path.exists(cls.model_path):
            return

        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_path, trust_remote_code=True)
        cls.model = AutoModelForCausalLM.from_pretrained(
            cls.model_path, 
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        ).cuda().eval()

    def setUp(self):
        if not torch.cuda.is_available() or not torch.os.path.exists(self.model_path):
            self.skipTest("GPU or Model weights not found. Skipping attention test.")

    def test_aomt_loss_is_lower_than_sft(self):
        """
        The core test from Engineering Spec section 11.4.
        Verifies that AOMT-Action-Only (with bidirectional attention) achieves a
        lower loss than Standard SFT (with causal attention) on the same data,
        proving that the model is correctly using future context.
        """
        print("\nStarting test: test_aomt_loss_is_lower_than_sft")
        
        # 1. Create a trajectory with a future cue that makes the middle action trivial to predict
        # Context: "You are in a room. To your left is a [MASK]. To your right is a table."
        # Action: "go left"
        # Future Cue: "You are now holding the golden apple that was on the left pedestal."
        
        obs0 = "You are in a room. To your left is a pedestal. To your right is a table."
        act0 = "go left"
        obs1 = "You are now holding the golden apple that was on the left pedestal."
        
        # 2. Run Forward Pass with Causal Mask (Standard SFT)
        # In SFT, model only sees obs0 to predict act0.
        messages = [{"role": "user", "content": obs0}, {"role": "assistant", "content": act0}]
        input_ids_sft = self.tokenizer.apply_chat_template(messages, return_tensors="pt").cuda()
        # Prompt length is everything before 'act0'
        prompt_text = self.tokenizer.apply_chat_template(messages[:1], add_generation_prompt=True, tokenize=False)
        prompt_len = len(self.tokenizer.encode(prompt_text))
        
        # Apply deterministic unit mask for SFT
        masked_ids_sft, labels_sft = apply_response_unit_mask(input_ids_sft.cpu(), torch.tensor([prompt_len]))
        masked_ids_sft = masked_ids_sft.cuda()
        labels_sft = labels_sft.cuda()
        
        # Construct CAUSAL mask (Standard SFT behavior)
        seq_len = input_ids_sft.shape[1]
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), device="cuda")).unsqueeze(0).unsqueeze(0)
        
        with torch.no_grad():
            logits_sft = self.model(input_ids=masked_ids_sft, attention_mask=causal_mask).logits
            loss_sft = F.cross_entropy(logits_sft.view(-1, logits_sft.size(-1)), labels_sft.view(-1), ignore_index=-100)

        # 3. Run Forward Pass with Bidirectional Mask (AOMT Action-Only)
        # In AOMT, model sees obs0 AND obs1 to predict act0.
        unit_texts = [obs0, act0, obs1]
        unit_types = ["obs", "act", "obs"]
        
        # Mock RNG to force mask only the 'act' unit
        class ForceMaskAct:
            def random(self): return 0.0 # Always mask
            def integers(self, n): return 0
            
        input_ids_aomt, labels_aomt = apply_unit_mask(
            unit_texts, unit_types, self.tokenizer, 
            mask_prob=1.0, mode="action_only", rng=ForceMaskAct()
        )
        input_ids_aomt = input_ids_aomt.unsqueeze(0).cuda()
        labels_aomt = labels_aomt.unsqueeze(0).cuda()
        
        # Full bidirectional mask (AOMT behavior)
        with torch.no_grad():
            logits_aomt = self.model(input_ids=input_ids_aomt).logits # Default is bidirectional
            loss_aomt = F.cross_entropy(logits_aomt.view(-1, logits_aomt.size(-1)), labels_aomt.view(-1), ignore_index=-100)

        print(f"  SFT Loss (Causal):    {loss_sft.item():.4f}")
        print(f"  AOMT Loss (Bidirect): {loss_aomt.item():.4f}")

        # The core assertion: AOMT loss should be lower because it sees the 'golden apple' future cue
        self.assertLess(loss_aomt, loss_sft, "AOMT loss should be lower than SFT loss when future context is available.")
        print("Test passed: Bidirectional attention correctly leverages future context.")

if __name__ == '__main__':
    unittest.main()
