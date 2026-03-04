import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import importlib
import yaml
import os
import numpy as np

# Lazy imports for environments
alfworld_env_module = None
scienceworld_env_module = None
webshop_env_module = None

def _lazy_import_environments():
    """Lazily imports environment packages."""
    global alfworld_env_module, scienceworld_env_module, webshop_env_module
    if alfworld_env_module is None:
        try:
            alfworld_env_module = importlib.import_module('alfworld.agents.environment')
        except ImportError:
            print("Warning: `alfworld` package not found. ALFWorld evaluation will be skipped.")
    if scienceworld_env_module is None:
        try:
            scienceworld_env_module = importlib.import_module('scienceworld')
        except ImportError:
            print("Warning: `scienceworld` package not found. ScienceWorld evaluation will be skipped.")
    if webshop_env_module is None:
        try:
            webshop_env_module = importlib.import_module('webshop.env')
        except ImportError:
            print("Warning: `webshop` package not found. WebShop evaluation will be skipped.")

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

def run_task_evaluation(model, tokenizer, env_name: str, eval_config: dict, split: str = "seen", device="cuda"):
    """
    High-level function to run task-based evaluation on real environments.
    """
    print(f"\n--- Running Task Evaluation on {env_name} ({split} split) ---")
    
    _lazy_import_environments() # Ensure environments are imported

    env = None
    tasks_to_evaluate = []
    
    if env_name == "alfworld" and alfworld_env_module:
        alf_conf = eval_config.get("alfworld", {})
        
        try:
            # AlfredTWEnv expects a dictionary as config
            # We'll construct a minimal one from eval_config
            config_for_env = {
                "alfworld": {
                    "data_path": alf_conf.get("data_path", "/tmp"), # Placeholder, should be set in eval_config
                    "task_filter": [] # Not using task_filter directly here, tasks come from 'tasks_to_evaluate'
                }
            }
            
            env = alfworld_env_module.AlfredTWEnv(config_for_env["alfworld"], train_eval=alf_conf.get("train_eval_split", "eval_out_of_distribution"))
            tasks_to_evaluate = alf_conf.get(f"{split}_tasks", [])
        except Exception as e:
            print(f"Error initializing ALFWorld environment: {e}. Skipping ALFWorld.")
            return {}

    elif env_name == "scienceworld" and scienceworld_env_module:
        sw_conf = eval_config.get("scienceworld", {})
        try:
            env = scienceworld_env_module.ScienceWorldEnv()
            tasks_data = sw_conf.get(f"{split}_tasks", [])
            
            for task_spec in tasks_data:
                env.load(task_spec["name"], task_spec["variation_idx"], sw_conf.get("simplification_str", "easy"))
                tasks_to_evaluate.append(env)

        except Exception as e:
            print(f"Error initializing ScienceWorld environment: {e}. Skipping ScienceWorld.")
            return {}

    elif env_name == "webshop" and webshop_env_module:
        ws_conf = eval_config.get("webshop", {})
        try:
            env = webshop_env_module.WebAgentTextEnv(
                observation_mode=ws_conf.get("observation_mode", "text"),
                human_goals=ws_conf.get("human_goals", True)
            )
            tasks_to_evaluate = ws_conf.get(f"{split}_tasks", [])

        except Exception as e:
            print(f"Error initializing WebShop environment: {e}. Skipping WebShop.")
            return {}
    
    if env is None or not tasks_to_evaluate:
        print(f"Warning: Environment '{env_name}' not properly initialized or no tasks to evaluate. Skipping.")
        return {}

    results = {"success_rate": [], "scores": []}
    max_episode_steps = eval_config.get("max_episode_steps", 100)

    for task_item in tqdm(tasks_to_evaluate, desc=f"Evaluating {env_name} tasks"):
        current_env = env 
        current_task_name = ""

        if env_name == "scienceworld":
            current_env = task_item 
            current_task_name = current_env.get_task_name()
        elif env_name == "alfworld":
            current_task_name = task_item
        elif env_name == "webshop":
            current_task_name = task_item

        if env_name == "alfworld":
            obs, info = current_env.reset({ 'task': current_task_name })
            obs_str = obs[0]
            task_success = 0.0
        elif env_name == "scienceworld":
            obs_str = current_env.reset() 
            task_success = 0.0
        elif env_name == "webshop":
            obs_str = current_env.reset(current_task_name)
            task_success = 0.0

        full_context_str = f"Observation: {obs_str}\n"
        done = False
        steps_taken = 0
        
        while not done and steps_taken < max_episode_steps:
            prompt_ids = tokenizer.encode(full_context_str, return_tensors="pt", max_length=model.config.max_position_embeddings - gen_length).to(device)
            
            action_text = llada_generate(model, tokenizer, prompt_ids, device=device)
            action_text = action_text.split("Act:")[1].strip() if "Act:" in action_text else action_text

            if env_name == "alfworld":
                obs_list, reward, done, info = current_env.step([action_text])
                obs_str = obs_list[0]
                is_success = info.get('won', False)
                task_success = 1.0 if is_success else 0.0
            elif env_name == "scienceworld":
                reward, done, obs_str = current_env.step(action_text)
                task_success = reward
            elif env_name == "webshop":
                obs_str, reward, done, info = current_env.step(action_text)
                task_success = reward
            
            full_context_str += f"Action: {action_text}\nObservation: {obs_str}\n"
            steps_taken += 1

        if env_name == "alfworld":
            results["success_rate"].append(task_success)
        elif env_name == "scienceworld":
            results["scores"].append(task_success) 
        elif env_name == "webshop":
            results["scores"].append(task_success)

    avg_success_rate = np.mean(results["success_rate"]) if results["success_rate"] else 0.0
    avg_score = np.mean(results["scores"]) if results["scores"] else 0.0

    output_metrics = {}
    if avg_success_rate > 0:
        output_metrics["avg_success_rate"] = avg_success_rate
    if avg_score > 0:
        output_metrics["avg_score"] = avg_score

    print(f"Evaluation on {env_name} complete. Results: {output_metrics}")
    return output_metrics