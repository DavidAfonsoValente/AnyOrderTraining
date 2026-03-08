# tests/test_attention_correctness.py
import unittest
import torch
import yaml
import os
import shutil
import numpy as np
from datasets import Dataset
from pathlib import Path
import itertools
from tqdm import tqdm

from ..training.trainer import AOMTDataset, AOMTDataCollator, training_step, BenchmarkUniformSampler
from ..training.mask_sampler import MaskMode
from veomni.models import build_tokenizer, build_foundation_model
from torch.utils.data import DataLoader

class TestAttentionMaskCorrectness(unittest.TestCase):

    def setUp(self):
        """
        Set up a minimal, controlled environment to test the attention mask logic.
        """
        self.test_dir = Path("test_attention_workspace")
        self.test_dir.mkdir(exist_ok=True)
        self.model_path = Path("./models/LLaDA2.0-mini")
        self.dummpy_model_config_path = self.test_dir / "model_config"
        self.dummpy_model_config_path.mkdir(exist_ok=True)

        if not self.model_path.exists():
            self.fail(
                f"Model files not found at {self.model_path}. "
                "This test requires the model to be downloaded first via './setup.sh'."
            )

        # Create a dummy model config directory for dFactory
        # In a real run, this would be 'dFactory/configs/model_configs/llada2_mini'
        (self.dummpy_model_config_path / "model.yaml").touch()


        # --- 1. Create two minimal config files (SFT vs AOMT-Action-Only) ---
        base_config = {
            'model': {
                'config_path': str(self.dummpy_model_config_path),
                'model_path': str(self.model_path),
                'tokenizer_path': str(self.model_path),
                'attn_implementation': 'sdpa',
            },
            'learning_rate': 1e-4,
            'weight_decay': 0.0,
            'gradient_clip': 1.0,
            'per_device_batch_size': 1,
            'num_epochs': 1, # This will be overridden by max_steps
            'train_split': 'train',
        }

        self.sft_config = base_config.copy()
        self.sft_config['name'] = 'sft_attention_test'
        self.sft_config['mask_mode'] = 'standard_sft'
        self.sft_config_path = self.test_dir / "sft_config.yaml"
        with open(self.sft_config_path, 'w') as f:
            yaml.dump(self.sft_config, f)

        self.aomt_config = base_config.copy()
        self.aomt_config['name'] = 'aomt_attention_test'
        self.aomt_config['mask_mode'] = 'action_only'
        self.aomt_config['mask_prob'] = 1.0 # Mask all actions
        self.aomt_config_path = self.test_dir / "aomt_config.yaml"
        with open(self.aomt_config_path, 'w') as f:
            yaml.dump(self.aomt_config, f)

        # --- 2. Create a small, fixed, and representative dataset ---
        # Trajectory: O0, A0, O1, A1. (4 units)
        self.dummy_dataset_path = self.test_dir / "processed_dataset"
        self.dummy_dataset_path.mkdir(exist_ok=True)
        
        # Manually create a single data point that is easy to reason about
        # O0: "go" (1 token), A0: "left" (1 token), O1: "see" (1 token), A1: "cat" (1 token)
        # Using simple words that are likely single tokens.
        tokenizer = build_tokenizer(str(self.model_path))
        input_ids = tokenizer.encode("go left see cat", add_special_tokens=False)
        
        dummy_data = {
            "id": ["dummy_0"],
            "env": ["test"],
            "input_ids": [input_ids],
            "unit_spans_type": [['obs', 'act', 'obs', 'act']],
            "unit_spans_start": [[0, 1, 2, 3]],
            "unit_spans_end": [[1, 2, 3, 4]],
        }
        arrow_dataset = Dataset.from_dict(dummy_data)
        arrow_dataset.save_to_disk(self.dummy_dataset_path)

    def tearDown(self):
        """Clean up the test workspace."""
        shutil.rmtree(self.test_dir)
        # Also clean up the checkpoints created by the trainer
        shutil.rmtree(f"./checkpoints/{self.sft_config['name']}", ignore_errors=True)
        shutil.rmtree(f"./checkpoints/{self.aomt_config['name']}", ignore_errors=True)

    def _run_mini_training(self, config_path: Path, max_steps=10) -> float:
        """
        A stripped-down, controlled training loop that runs for a few steps
        on a fixed dataset and returns the final loss.
        This avoids using the full `run_training` function to have more control.
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        tokenizer = build_tokenizer(config['model']['tokenizer_path'])
        if tokenizer.pad_token is None: tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        if tokenizer.mask_token is None: tokenizer.add_special_tokens({'mask_token': '[MASK]'})

        processed_dataset = Dataset.load_from_disk(self.dummy_dataset_path)
        
        mask_mode = MaskMode(config["mask_mode"])
        dataset = AOMTDataset(
            processed_dataset, tokenizer, mode=mask_mode,
            mask_prob=config.get("mask_prob", 0.0), seed=42
        )
        # Use a deterministic sampler (no shuffling)
        sampler = torch.utils.data.SequentialSampler(dataset)
        
        collator = AOMTDataCollator(tokenizer)
        # Set shuffle=False to ensure data is identical for both runs
        dataloader = DataLoader(dataset, batch_size=1, collate_fn=collator, sampler=sampler)

        model = build_foundation_model(
            config_path=config["model"]["config_path"],
            weights_path=config["model"]["model_path"],
            torch_dtype=torch.float32, # Use float32 for deterministic testing
            attn_implementation="eager", # Use eager for determinism
        )
        model.resize_token_embeddings(len(tokenizer))
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
        
        model.train()
        losses = []
        dataloader_iterator = itertools.cycle(dataloader)
        
        print(f"\n--- Running mini-training for {config['name']} ---")
        for step in tqdm(range(max_steps), desc=f"Training {config['name']}"):
            batch = next(dataloader_iterator)
            
            loss = training_step(model, batch, "cuda" if torch.cuda.is_available() else "cpu")
            
            if loss.isnan() or loss.isinf():
                self.fail("Loss became NaN or Inf during training.")

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
        
        return np.mean(losses[-3:]) # Return the average of the last few losses

    def test_aomt_loss_is_lower_than_sft(self):
        """
        The core test from Engineering Spec section 11.4.
        Verifies that AOMT-Action-Only (with bidirectional attention) achieves a
        lower loss than Standard SFT (with causal attention) on the same data,
        proving that the model is correctly using future context.
        """
        print("\n" + "="*50)
        print("Starting test: test_aomt_loss_is_lower_than_sft")
        print("="*50)

        # Ensure CUDA is available for this test to be meaningful
        if not torch.cuda.is_available():
            self.skipTest("This test requires a GPU to run.")

        # Run training for SFT
        final_loss_sft = self._run_mini_training(self.sft_config_path)
        print(f"Final average loss for SFT: {final_loss_sft:.4f}")

        # Run training for AOMT
        final_loss_aomt = self._run_mini_training(self.aomt_config_path)
        print(f"Final average loss for AOMT (Action-Only): {final_loss_aomt:.4f}")

        # The core assertion
        self.assertLess(
            final_loss_aomt,
            final_loss_sft,
            "AOMT loss was NOT lower than SFT loss. This indicates the "
            "bidirectional attention mask is likely not being applied correctly."
        )
        
        print("\nTest passed: AOMT loss was significantly lower than SFT loss.")
        print(f"  AOMT: {final_loss_aomt:.4f} < SFT: {final_loss_sft:.4f}")


if __name__ == '__main__':
    unittest.main()
