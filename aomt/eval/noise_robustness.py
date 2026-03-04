# aomt/eval/noise_robustness.py
import numpy as np
from transformers import AutoTokenizer

from aomt.eval.task_eval import run_task_evaluation, MockEnv

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

    Args:
        text (str): The original observation text.
        tokenizer (AutoTokenizer): The tokenizer to use.
        corruption_frac (float): The fraction of tokens to corrupt (e.g., 0.1 for 10%).
        rng (np.random.Generator): A random number generator for reproducibility.

    Returns:
        str: The corrupted observation text.
    """
    if corruption_frac == 0:
        return text

    # Encode the text but don't add special tokens
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        return ""

    # Determine the number of tokens to corrupt
    n_tokens = len(ids)
    n_corrupt = max(1, int(n_tokens * corruption_frac))
    
    # Choose random positions to corrupt
    # Ensure we don't try to choose more tokens than available
    n_corrupt = min(n_corrupt, n_tokens)
    positions_to_corrupt = rng.choice(n_tokens, size=n_corrupt, replace=False)

    # Replace tokens at those positions with random vocabulary tokens
    # We avoid replacing with special tokens by sampling from a safe range.
    for pos in positions_to_corrupt:
        random_token_id = int(rng.integers(100, tokenizer.vocab_size))
        ids[pos] = random_token_id

    # Decode back to text
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
        """
        Takes an action, gets the next observation, corrupts it, and returns it.
        """
        obs, reward, done = self.env.step(action)
        corrupted_obs = corrupt_observation(
            obs, self.tokenizer, self.corruption_frac, self.rng
        )
        return corrupted_obs, reward, done

    def reset(self):
        """
        Resets the environment and corrupts the initial observation.
        """
        obs = self.env.reset()
        corrupted_obs = corrupt_observation(
            obs, self.tokenizer, self.corruption_frac, self.rng
        )
        return corrupted_obs

def run_noise_robustness_evaluation(model, tokenizer, env_name, tasks, device="cuda"):
    """
    Runs task evaluation across multiple levels of observation noise.
    """
    noise_fractions = [0.0, 0.1, 0.2, 0.3] # As specified in the eng specs
    results = {}

    for frac in noise_fractions:
        print(f"\n--- Evaluating with Noise Fraction: {frac:.1f} ---")
        
        # Here you would wrap a real environment
        # For demonstration, we wrap the mock environment
        base_env = MockEnv()
        
        if frac > 0:
            env = CorruptedEnvWrapper(base_env, tokenizer, corruption_frac=frac)
        else:
            env = base_env # No corruption for the baseline run

        # We need a way to pass the custom environment to the evaluator
        # For now, let's adapt the logic from run_task_evaluation here.
        # (This suggests run_task_evaluation could be refactored to accept an env object)
        
        success_rates = []
        for task in tasks:
            base_env.current_observation = f"{env_name.title()} Task: {task}"
            obs = env.reset()
            full_context_str = f"Observation: {obs}\n"
            done = False
            
            while not done:
                prompt_ids = tokenizer.encode(full_context_str, return_tensors="pt")
                action_text = "default action" # Placeholder call
                obs, reward, done = env.step(action_text)
                full_context_str += f"Action: {action_text}\nObservation: {obs}\n"
            success_rates.append(reward)

        avg_success = np.mean(success_rates)
        print(f"Noise {frac:.1f}: Average Success = {avg_success:.2%}")
        results[f"success_rate_rho{int(frac*10)}"] = avg_success
        
    return results


if __name__ == '__main__':
    # This is a demonstration. It requires a model and tokenizer.
    import os
    import importlib
    from transformers import AutoModelForCausalLM
    
    model_path = "./models/LLaDA2.0-mini"
    if importlib.util.find_spec("transformers") and os.path.isdir(model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        # model = AutoModelForCausalLM.from_pretrained(model_path)
        
        print("Demonstrating observation corruption:")
        rng = np.random.default_rng(42)
        original_text = "You are in a kitchen. You see a fridge, a countertop, and a sink."
        corrupted_text = corrupt_observation(original_text, tokenizer, 0.2, rng)
        print(f"Original:    {original_text}")
        print(f"Corrupted:   {corrupted_text}")

        # The full evaluation would be run like this:
        # mock_tasks = ["put hot potato on countertop", "find the key"]
        # run_noise_robustness_evaluation(model, tokenizer, "alfworld", mock_tasks)
        print("\nSkipping full noise robustness evaluation loop in this example.")
    else:
        print("Skipping noise_robustness demonstration.")
        print("Please ensure `transformers` is installed and the model is downloaded to ./models/LLaDA2.0-mini")
