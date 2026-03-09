# tests/test_training_integration.py
import unittest
import torch
import yaml
import os
import sys
import shutil
import json
import torch.distributed as dist

from tasks.train_aomt import main as run_training

class TestTrainingIntegration(unittest.TestCase):

    def setUp(self):
        """
        Set up a minimal training environment for a single-step integration test.
        """
        self.test_dir = "test_integration_workspace"
        os.makedirs(self.test_dir, exist_ok=True)
        os.makedirs(os.path.join(self.test_dir, "data"), exist_ok=True)

        # --- 1. Create a dummy small model ---
        self.dummy_model_dir = os.path.join(self.test_dir, "dummy_model")
        os.makedirs(self.dummy_model_dir, exist_ok=True)
        
        # Use a small config to avoid OOM
        config_dict = {
            "architectures": ["LLaDA2MoeModelLM"],
            "auto_map": {
                "AutoConfig": "configuration_llada2_moe.LLaDA2MoeConfig",
                "AutoModel": "modeling_llada2_moe.LLaDA2MoeModel",
                "AutoModelForCausalLM": "modeling_llada2_moe.LLaDA2MoeModelLM"
            },
            "model_type": "llada2_moe",
            "num_hidden_layers": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_experts": 2,
            "num_experts_per_tok": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "moe_intermediate_size": 32,
            "vocab_size": 157184,  # Match tokenizer
            "hidden_act": "silu",
            "head_dim": 16,
            "num_shared_experts": 1,
            "max_position_embeddings": 512,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": False,
            "rope_theta": 600000,
            "partial_rotary_factor": 0.5,
            "rotary_dim": 16,
            "n_group": 1,
            "topk_group": 1,
            "norm_topk_prob": True,
            "router_dtype": "fp32",
            "score_function": "sigmoid",
            "moe_router_enable_expert_bias": True,
            "routed_scaling_factor": 1.0,
        }
        with open(os.path.join(self.dummy_model_dir, "config.json"), "w") as f:
            json.dump(config_dict, f)
            
        # Copy modeling files from existing model or dFactory
        src_modeling = "weights/LLaDA2.0-mini/modeling_llada2_moe.py"
        src_config = "weights/LLaDA2.0-mini/configuration_llada2_moe.py"
        if not os.path.exists(src_modeling):
            src_modeling = "dFactory/models/llada2_moe/modeling_llada2_moe.py"
            src_config = "dFactory/models/llada2_moe/configuration_llada2_moe.py"
            
        if os.path.exists(src_modeling):
            shutil.copy(src_modeling, self.dummy_model_dir)
            shutil.copy(src_config, self.dummy_model_dir)
        else:
            self.skipTest(f"Modeling files not found. Skipping integration test.")

        # Create an empty weights file to satisfy the loader
        torch.save({}, os.path.join(self.dummy_model_dir, "pytorch_model.bin"))

        # Create a dummy JSONL file
        self.jsonl_path = os.path.join(self.test_dir, "data", "dummy.jsonl")
        dummy_entry = {
            "unit_texts": ["Observation 1", "Action 1", "Observation 2"],
            "unit_types": ["obs", "act", "obs"]
        }
        with open(self.jsonl_path, "w") as f:
            f.write(json.dumps(dummy_entry) + "\n")

        # --- 2. Create a minimal config file ---
        self.config = {
            'name': 'integration_test',
            'model': {
                'model_path': self.dummy_model_dir,
                'tokenizer_path': 'weights/LLaDA2.0-mini',
            },
            'data': {
                'train_path': self.jsonl_path,
            },
            'aomt': {
                'mode': 'mixed',
                'mask_prob': 0.25,
            },
            'train': {
                'learning_rate': 1e-5,
                'weight_decay': 0.1,
                'gradient_clip': 1.0,
                'per_device_batch_size': 1,
                'num_epochs': 1,
                'mixed_precision': 'bf16', 
                'lr_scheduler': 'cosine',
                'warmup_steps': 0,
                'output_dir': os.path.join(self.test_dir, "output"),
                'gradient_checkpointing': False, # Disable for tiny model
            }
        }
        self.config_path = os.path.join(self.test_dir, "test_config.yaml")
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f)

        # --- 3. Check for the required tokenizer ---
        if not os.path.exists(self.config['model']['tokenizer_path']):
            self.skipTest(f"Tokenizer not found at {self.config['model']['tokenizer_path']}. Skipping integration test.")

        # --- 4. Mock Distributed Env for single-process test ---
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29505"
        os.environ["LOCAL_RANK"] = "0"

    def tearDown(self):
        """Clean up the test workspace."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        if dist.is_initialized():
            dist.destroy_process_group()

    def test_single_step_run(self):
        """
        Sanity Check 11.4: Verify that the full training pipeline can execute.
        """
        print("\nRunning test: test_single_step_run")
        
        # Patch sys.argv to pass the config path to run_training
        old_argv = sys.argv
        sys.argv = [old_argv[0], self.config_path]
        
        try:
            # Run the task's training loop
            run_training()
            self.assertTrue(True)
            print("Test passed: Single training step completed successfully.")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.fail(f"Training integration test failed with an exception: {e}")
        finally:
            sys.argv = old_argv

if __name__ == '__main__':
    unittest.main()
