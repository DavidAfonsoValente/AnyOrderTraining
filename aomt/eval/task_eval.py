# aomt/eval/task_eval.py
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import importlib

# Placeholder for actual environment interaction.
# In a real setup, this would be `alfworld.agents.environment` or similar.
class MockEnv:
    def __init__(self, initial_observation="Initial state."):
        self.current_observation = initial_observation
        self.steps = 0
        self.max_steps = 10
        self.done = False

    def step(self, action: str):
        if self.done:
            return self.current_observation, 0, self.done
        
        print(f"Action: {action}")
        self.steps += 1
        if "success" in action or self.steps >= self.max_steps:
            self.done = True
            reward = 1.0 if "success" in action else 0.0
            self.current_observation = "Task finished."
        else:
            self.current_observation = f"You took action '{action}'. State updated."
            reward = 0.0
        
        print(f"Observation: {self.current_observation}")
        return self.current_observation, reward, self.done

    def reset(self):
        self.steps = 0
        self.done = False
        self.current_observation = "Initial state."
        return self.current_observation

# --- Section 8.2: LLaDA 2.0 Inference ---

@torch.no_grad()
def llada_generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_ids: torch.Tensor,
    gen_length: int = 128,
    block_length: int = 32,
    steps: int = 32,
    temperature: float = 0.0,
    device: str = "cuda"
) -> str:
    """
    Generates an action using LLaDA 2.0's iterative unmasking (block diffusion).

    At inference, the model generates an entire action from a fully masked
    template, progressively refining its prediction over several steps.

    Args:
        model: The trained model.
        tokenizer: The tokenizer.
        prompt_ids (torch.Tensor): The tokenized context (observation history). Shape: [1, context_len].
        gen_length (int): The number of tokens to generate for the action.
        block_length (int): LLaDA-specific generation parameter.
        steps (int): Number of refinement steps for generation.
        temperature (float): Sampling temperature. 0.0 for greedy.
        device (str): The device to run generation on.

    Returns:
        str: The decoded text of the generated action.
    """
    model.eval()
    model.to(device)

    mask_token_id = tokenizer.mask_token_id
    prompt_ids = prompt_ids.to(device)
    prompt_len = prompt_ids.shape[1]

    # 1. Start with a fully masked response
    masked_response = torch.full((1, gen_length), mask_token_id, dtype=torch.long, device=device)
    full_input = torch.cat([prompt_ids, masked_response], dim=1)

    # 2. Iteratively unmask tokens over `steps`
    for step_idx in range(steps):
        logits = model(input_ids=full_input).logits

        # Focus on the logits for the response part
        response_logits = logits[:, prompt_len:, :]
        
        # Sample tokens from the distribution
        if temperature > 0:
            probs = torch.softmax(response_logits / temperature, dim=-1)
            pred_tokens = torch.multinomial(probs.squeeze(0), num_samples=1).squeeze(-1)
        else: # Greedy decoding
            pred_tokens = response_logits.argmax(dim=-1)
        
        pred_tokens = pred_tokens.squeeze(0)
        
        # Calculate confidence (probability of the predicted token)
        probs = torch.softmax(response_logits.squeeze(0), dim=-1)
        confidence = probs.gather(1, pred_tokens.unsqueeze(1)).squeeze(1)

        # Identify which tokens are still masked
        is_masked = (full_input[0, prompt_len:] == mask_token_id)
        
        # Replace predicted tokens in the input for the next iteration
        full_input[0, prompt_len:] = torch.where(is_masked, pred_tokens, full_input[0, prompt_len:])

        # Determine how many tokens to "commit" to (unmask) in this step
        # This is a simplified version of the original paper's masking schedule
        num_masked = is_masked.sum()
        num_to_unmask = int( (1 / (steps - step_idx)) * num_masked ) if steps != step_idx else num_masked

        # Find the least confident tokens among the currently predicted ones
        # and re-mask them for the next iteration.
        inv_confidence = 1.0 - confidence
        inv_confidence[~is_masked] = -1 # Don't re-mask already committed tokens
        
        _, remask_indices = torch.topk(inv_confidence, k=int(num_masked - num_to_unmask))
        
        full_input[0, prompt_len:][remask_indices] = mask_token_id

    # 3. Decode the final generated sequence
    final_response_ids = full_input[0, prompt_len:]
    return tokenizer.decode(final_response_ids, skip_special_tokens=True)

def run_task_evaluation(model, tokenizer, env_name: str, tasks: list, device="cuda"):
    """
    High-level function to run task-based evaluation.
    """
    print(f"\n--- Running Task Evaluation on {env_name} ---")
    
    try:
        if env_name == "alfworld":
            # Lazy import alfworld
            import alfworld.agents.environment
        # Add other envs here
    except ImportError:
        print(f"Environment '{env_name}' not installed. Using MockEnv.")
        env_name = "mock"

    results = {"success_rate": [], "scores": []}

    for task in tqdm(tasks, desc=f"Evaluating {env_name}"):
        if env_name == "alfworld":
            # This is complex; requires a config and task files.
            # For this implementation, we will use a mock.
            env = MockEnv(initial_observation=f"ALFWorld Task: {task}")
        else:
            env = MockEnv(initial_observation=f"Task: {task}")
            
        obs = env.reset()
        full_context_str = f"Observation: {obs}\n"
        done = False
        
        while not done:
            prompt_ids = tokenizer.encode(full_context_str, return_tensors="pt")
            
            action_text = llada_generate(model, tokenizer, prompt_ids, device=device)
            action_text = action_text.split("Act:")[1].strip() if "Act:" in action_text else action_text

            obs, reward, done = env.step(action_text)

            # Append to context
            full_context_str += f"Action: {action_text}\nObservation: {obs}\n"

        results["success_rate"].append(reward) # For binary success

    avg_success = np.mean(results["success_rate"])
    print(f"Average Success Rate on {env_name}: {avg_success:.2%}")
    return {"avg_success_rate": avg_success}

if __name__ == '__main__':
    # This is a demonstration of how to use the functions.
    # It requires a trained model and tokenizer.
    model_path = "./models/LLaDA2.0-mini"
    if not importlib.util.find_spec("alfworld"):
         print("Warning: alfworld is not installed. Task evaluation will use a mock environment.")
         
    if importlib.util.find_spec("transformers") and os.path.isdir(model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        
        # Example tasks for a mock evaluation
        mock_tasks = ["put hot potato on countertop", "find the key"]
        run_task_evaluation(model, tokenizer, "alfworld", mock_tasks)
    else:
        print("Skipping task_eval demonstration.")
        print("Please ensure `transformers` is installed and the model is downloaded to ./models/LLaDA2.0-mini")
