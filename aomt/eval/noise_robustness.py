# aomt/eval/noise_robustness.py
import numpy as np
from transformers import AutoTokenizer
import torch

from aomt.eval.task_eval import llada_generate, _lazy_import_environments, alfworld_env_module, scienceworld_env_module, webshop_env_module

# --- Section 8.3: Robustness Evaluation ---

def corrupt_observation(
    text: str,
    tokenizer: AutoTokenizer,
    corruption_frac: float,
    rng: np.random.Generator,
) -> str:
    """
    Corrupts an observation by replacing a fraction of its tokens with random
    tokens from the tokenizer's vocabulary.
    """
    if corruption_frac == 0:
        return text

    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        return ""

    n_tokens = len(ids)
    n_corrupt = max(1, int(n_tokens * corruption_frac))
    n_corrupt = min(n_corrupt, n_tokens)
    positions_to_corrupt = rng.choice(n_tokens, size=n_corrupt, replace=False)

    for pos in positions_to_corrupt:
        random_token_id = int(rng.integers(100, tokenizer.vocab_size))
        ids[pos] = random_token_id

    return tokenizer.decode(ids, skip_special_tokens=True)


class CorruptedEnvWrapper:
    """
    A wrapper around an environment that corrupts observations before returning them.
    """
    def __init__(
        self,
        env,
        tokenizer: AutoTokenizer,
        corruption_frac: float,
        seed: int = 42
    ):
        self.env = env
        self.tokenizer = tokenizer
        self.corruption_frac = corruption_frac
        self.rng = np.random.default_rng(seed)

    def step(self, action: str):
        obs, reward, done, info = self.env.step(action)
        corrupted_obs = [corrupt_observation(o, self.tokenizer, self.corruption_frac, self.rng) for o in obs]
        return corrupted_obs, reward, done, info

    def reset(self, task_info=None):
        if task_info:
            obs, info = self.env.reset(task_info)
        else:
            obs, info = self.env.reset()
        
        corrupted_obs = [corrupt_observation(o, self.tokenizer, self.corruption_frac, self.rng) for o in obs]
        return corrupted_obs, info

def run_noise_robustness_evaluation(model, tokenizer, env_name, eval_config: dict, split: str = "seen", device="cuda"):
    """
    Runs task evaluation across multiple levels of observation noise.
    """
    noise_fractions = [0.0, 0.1, 0.2, 0.3]
    results = {}

    for frac in noise_fractions:
        print(f"\n--- Evaluating {env_name} with Noise Fraction: {frac:.1f} ---")
        
        _lazy_import_environments()
        
        base_env = None
        if env_name == "alfworld" and alfworld_env_module:
            alf_conf = eval_config.get("alfworld", {})
            base_env = alfworld_env_module.AlfredTWEnv(alf_conf, train_eval=alf_conf.get("train_eval_split", "eval_out_of_distribution"))
        # Add other envs here
        
        if base_env is None:
            print(f"Could not initialize base environment for {env_name}. Skipping noise evaluation.")
            continue

        env = CorruptedEnvWrapper(base_env, tokenizer, corruption_frac=frac)
        
        tasks_to_evaluate = eval_config.get(env_name, {}).get(f"{split}_tasks", [])
        if not tasks_to_evaluate:
            print(f"No tasks found for {env_name} {split} split. Skipping.")
            continue
            
        success_rates = []
        for task in tasks_to_evaluate:
            obs, info = env.reset({'task': task})
            full_context_str = f"Observation: {obs[0]}\n"
            done = False
            
            while not done:
                prompt_ids = tokenizer.encode(full_context_str, return_tensors="pt").to(device)
                action_text = llada_generate(model, tokenizer, prompt_ids, device=device)
                action_text = action_text.split("Act:")[1].strip() if "Act:" in action_text else action_text
                
                obs, reward, done, info = env.step([action_text])
                full_context_str += f"Action: {action_text}\nObservation: {obs[0]}\n"
            
            success_rates.append(1.0 if info.get('won', False) else 0.0)

        avg_success = np.mean(success_rates)
        print(f"Noise {frac:.1f}: Average Success = {avg_success:.2%}")
        results[f"success_rate_rho{int(frac*10)}"] = avg_success
        
    return results
