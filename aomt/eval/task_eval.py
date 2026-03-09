"""
eval/task_eval.py
Full final evaluation script for ALFWorld, ScienceWorld, and WebShop.
Implements real environment interaction loops using standard benchmark APIs.
No mock results; handles real trajectories, history management, and LLaDA 2.0 generation.
"""

import argparse
import json
import os
import torch
import numpy as np
import yaml
from tqdm import tqdm
from transformers import AutoTokenizer

# Use local dFactory/VeOmni if available
try:
    from veomni.models import build_tokenizer, build_foundation_model
except ImportError:
    # Minimal fallbacks
    def build_tokenizer(path): return AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    def build_foundation_model(**kwargs): 
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(kwargs['weights_path'], trust_remote_code=True)

SYSTEM_PROMPT = """Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. 
For each of your turn, you will be given the observation of the last turn. You should first think about the current condition and plan for your future actions, and then output your action in this turn. Your output must strictly follow this format:"Thought: your thoughts.\\nAction: your next action".

The available actions are:
1. go to {recep}
2. task {obj} from {recep}
3. put {obj} in/on {recep}
4. open {recep}
5. close {recep}
6. toggle {obj} {recep}
7. clean {obj} with {recep}
8. heat {obj} with {recep}
9. cool {obj} with {recep}
where {obj} and {recep} correspond to objects and receptacles.
After your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the envrionment output "Nothing happened", that means the previous action is invalid and you should try more options.

Your response should use the following format:

Thought: <your thoughts>
Action: <your next action>"""

# -----------------------------------------------------------------------------
# Environment Interaction Functions
# -----------------------------------------------------------------------------

def evaluate_scienceworld(model, tokenizer, split="test", n_episodes=50, max_steps=100, **kwargs):
    """
    Evaluates ScienceWorld and returns average normalized score.
    Ref: https://github.com/allenai/scienceworld
    """
    from scienceworld import ScienceWorldEnv
    # Initialize the environment
    env = ScienceWorldEnv("", "", envStepLimit=max_steps)
    task_names = env.getTaskNames()
    
    scores = []
    print(f"Running ScienceWorld evaluation ({split} split, {n_episodes} episodes)...")
    
    for episode in tqdm(range(n_episodes)):
        # Task selection logic: cycle through tasks
        task_name = task_names[episode % len(task_names)]
        
        # Determine available variations for the split
        if split in ["test", "unseen"]:
            variations = env.getVariationsTest()
        elif split in ["dev", "val"]:
            variations = env.getVariationsDev()
        else: # Default to train or seen
            variations = env.getVariationsTrain()
            
        if not variations:
            print(f"Warning: No variations found for task {task_name} in split {split}. Skipping.")
            continue
            
        # Select variation based on episode count
        variation_idx = variations[(episode // len(task_names)) % len(variations)]
        
        # Load specific task/variation and reset
        env.load(task_name, variation_idx)
        observation, info = env.reset()
        
        # Initialize history with system prompt and assistant "OK"
        obs_history = SYSTEM_PROMPT + "\nOK\n" + observation
        done = False
        step = 0
        
        while not done and step < max_steps:
            # Predict the next action using LLaDA 2.0
            action = generate_action(model, tokenizer, obs_history, **kwargs)
            observation, reward, done, info = env.step(action)
            
            # Simple linear history concatenation
            obs_history += f"\n{action}\n{observation}"
            step += 1
            
        # Score is returned as 0-100; normalize to 0-1
        scores.append(info.get('score', 0.0) / 100.0)
    
    env.close()
    return {"avg_score": float(np.mean(scores)) if scores else 0.0, "episodes": len(scores)}

def evaluate_alfworld(model, tokenizer, split="test", n_episodes=50, max_steps=50, **kwargs):
    """
    Evaluates ALFWorld and returns success rate.
    Ref: https://github.com/alfworld/alfworld
    """
    import alfworld.agents.environment
    
    # ALFWorld requires a YAML config file for initialization
    config_path = os.environ.get("ALFWORLD_CONFIG", "configs/alfworld_config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"ALFWorld config not found at {config_path}. Check $ALFWORLD_CONFIG.")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Environment initialization (standard ALFWorld setup)
    print(f"Running ALFWorld evaluation ({split} split, {n_episodes} episodes)...")
    env = alfworld.agents.environment.get_environment(config)
    
    successes = []
    count = 0
    
    # ALFWorld env acts as a generator/iterator of episodes
    for env_session in env:
        if count >= n_episodes:
            break
            
        observation, info = env_session.reset()
        # Handle observation list (standard ALFWorld returns a list for batching)
        if isinstance(observation, list): observation = observation[0]
        
        # Initialize history with system prompt and assistant "OK"
        obs_history = SYSTEM_PROMPT + "\nOK\n" + observation
        done = False
        step = 0
        
        while not done and step < max_steps:
            # Predict action from current history
            action = generate_action(model, tokenizer, obs_history, **kwargs)
            
            # Step the environment (expects list of actions)
            observation, reward, done, info = env_session.step([action])
            
            # Unpack results
            if isinstance(observation, list): observation = observation[0]
            if isinstance(reward, list): reward = reward[0]
            if isinstance(done, list): done = done[0]
            
            obs_history += f"\n{action}\n{observation}"
            step += 1
            
        # Success is defined by binary reward in ALFWorld
        successes.append(1.0 if reward > 0 else 0.0)
        count += 1
        
    return {"success_rate": float(np.mean(successes)) if successes else 0.0, "episodes": len(successes)}

def evaluate_webshop(model, tokenizer, split="test", n_episodes=50, max_steps=10, **kwargs):
    """
    Evaluates WebShop and returns average reward.
    Automatically starts the local WebShop server if not active.
    Ref: https://github.com/princeton-nlp/WebShop
    """
    import subprocess
    import time
    import requests

    # 1. Ensure the WebShop server is running
    webshop_url = "http://localhost:3000" # Default WebShop port
    try:
        requests.get(webshop_url, timeout=2)
        print("WebShop server is already running.")
    except (requests.ConnectionError, requests.Timeout):
        print("WebShop server not detected. Starting it now...")
        webshop_path = os.path.join(os.path.dirname(__file__), 'WebShop')
        # Start the production server in the background
        subprocess.Popen(["bash", "run_prod.sh"], cwd=webshop_path, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Wait for server to boot
        max_retries = 30
        for i in range(max_retries):
            try:
                requests.get(webshop_url, timeout=2)
                print("WebShop server started successfully.")
                break
            except:
                if i % 5 == 0: print(f"  Waiting for server... ({i}/{max_retries})")
                time.sleep(2)
        else:
            raise RuntimeError("Failed to start WebShop server. Please run 'bash aomt/eval/WebShop/run_prod.sh' manually.")

    # 2. Initialize the environment
    try:
        from webshop.env import WebAgentTextEnv
    except ImportError:
        # Add WebShop to path if not installed as a package
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'WebShop'))
        from web_agent_site.envs import WebAgentTextEnv
        
    # Initialize the environment (standard WebShop setup)
    # n_episodes refers to the number of human goals to evaluate
    env = WebAgentTextEnv(observation_mode="text", split=split)
    
    rewards = []
    print(f"Running WebShop evaluation ({split} split, {n_episodes} episodes)...")
    
    for episode_idx in tqdm(range(n_episodes)):
        # Reset with specific human goal index
        observation, info = env.reset(episode_idx)
        
        # Initialize history with system prompt and assistant "OK"
        obs_history = SYSTEM_PROMPT + "\nOK\n" + observation
        done = False
        step = 0
        
        while not done and step < max_steps:
            # Predict next navigation action
            action = generate_action(model, tokenizer, obs_history, **kwargs)
            observation, reward, done, info = env.step(action)
            
            obs_history += f"\n{action}\n{observation}"
            step += 1
            
        # WebShop rewards are normalized 0-1 based on instruction-product alignment
        rewards.append(reward)
        
    return {"avg_reward": float(np.mean(rewards)) if rewards else 0.0, "episodes": len(rewards)}

# -----------------------------------------------------------------------------
# Generation Logic (LLaDA 2.0 specific)
# -----------------------------------------------------------------------------

def generate_action(model, tokenizer, obs_history_text: str, gen_length: int = 256, block_length: int = 32, steps: int = 32) -> str:
    """
    LLaDA 2.0 generation API.
    Utilizes bidirectional unmasking via the specialized model.generate call.
    Parses history into turns to match chat template expectations.
    """
    # Split history into turns. Training used strictly alternating format.
    # The history starts with SYSTEM_PROMPT, then OK, then observations/actions.
    # We reconstruct the message list.
    messages = []
    
    # Check if history starts with system prompt
    if obs_history_text.startswith("Interact with a household"):
        # Extract system prompt and OK
        parts = obs_history_text.split("\nOK\n", 1)
        messages.append({"role": "user", "content": parts[0].strip()})
        messages.append({"role": "assistant", "content": "OK"})
        remaining = parts[1] if len(parts) > 1 else ""
    else:
        remaining = obs_history_text

    # Split the rest by newline and try to identify turns.
    # Since actions are typically generated by assistant and observations by user.
    # In task_eval, we append history as: history += f"\n{action}\n{observation}"
    # This means everything after the initial "OK" alternates: Obs, Act, Obs, Act...
    
    turns = [t.strip() for t in remaining.split("\n") if t.strip()]
    for i, turn in enumerate(turns):
        role = "user" if i % 2 == 0 else "assistant"
        messages.append({"role": role, "content": turn})

    # Tokenize history using the chat template (Standard in dFactory/VeOmni)
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    ).to(model.device)

    # Context window management
    max_context = getattr(model.config, "max_position_embeddings", 2048) - gen_length - 8 
    if input_ids.shape[1] > max_context: 
        input_ids = input_ids[:, -max_context:]

    with torch.no_grad():
        # The LLaDA2MoeModelLM.generate method uses these specific names:
        output_ids = model.generate(
            inputs=input_ids,
            gen_length=gen_length,
            block_length=block_length,
            steps=steps,
            temperature=0.0,
            eos_early_stop=True,
            mask_id=tokenizer.mask_token_id or 156895
        )

    # Extract the generated response only
    generated = output_ids[0, input_ids.shape[1]:]
    
    # Post-hoc EOS truncation if needed
    eos_id = tokenizer.eos_token_id
    if eos_id is not None:
        eos_positions = (generated == eos_id).nonzero(as_tuple=True)[0]
        if len(eos_positions) > 0:
            generated = generated[:eos_positions[0]]

    # Decode and return stripped string
    full_response = tokenizer.decode(generated, skip_special_tokens=True).strip()
    
    # Standard ReAct parsing: Extract the action after "Action:"
    if "Action:" in full_response:
        action = full_response.split("Action:")[-1].strip()
    elif "action:" in full_response:
        action = full_response.split("action:")[-1].strip()
    else:
        # Fallback to the last line if no explicit "Action:" tag is found
        lines = [l.strip() for l in full_response.split("\n") if l.strip()]
        action = lines[-1] if lines else full_response

    return action

# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="AOMT Task Evaluation Suite")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--tokenizer", type=str, required=True, help="Path to the tokenizer")
    parser.add_argument("--benchmark", type=str, required=True, choices=["alfworld", "scienceworld", "webshop"], 
                        help="Benchmark to evaluate")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test", "seen", "unseen", "dev", "val"], 
                        help="Data split to evaluate on")
    parser.add_argument("--n_episodes", type=int, default=50, help="Number of episodes to evaluate")
    parser.add_argument("--gen_length", type=int, default=256, help="Max length of generated action")
    parser.add_argument("--block_length", type=int, default=32, help="Block length for LLaDA generation")
    parser.add_argument("--steps", type=int, default=32, help="Denoising steps per block")
    parser.add_argument("--output_file", type=str, required=True, help="JSON file to save results")
    args = parser.parse_args()

    # Hardware detection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}...")
    
    # Load model and tokenizer via VeOmni builders or HF fallbacks
    tokenizer = build_tokenizer(args.tokenizer)
    model = build_foundation_model(
        weights_path=args.model_path,
        config_path=args.model_path,
        torch_dtype="bfloat16",
        attn_implementation="sdpa",
        init_device=device
    )
    model.to(device)
    model.eval()

    # Bundle generation parameters
    gen_params = {
        "gen_length": args.gen_length,
        "block_length": args.block_length,
        "steps": args.steps
    }

    # Execute the selected benchmark evaluation
    if args.benchmark == "scienceworld":
        result = evaluate_scienceworld(model, tokenizer, split=args.split, n_episodes=args.n_episodes, **gen_params)
    elif args.benchmark == "alfworld":
        result = evaluate_alfworld(model, tokenizer, split=args.split, n_episodes=args.n_episodes, **gen_params)
    elif args.benchmark == "webshop":
        result = evaluate_webshop(model, tokenizer, split=args.split, n_episodes=args.n_episodes, **gen_params)

    # Final summary output
    print(f"\nEvaluation Complete for {args.benchmark} ({args.split})")
    print(json.dumps(result, indent=4))
    
    # Persist results to disk
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(result, f, indent=4)

if __name__ == "__main__":
    main()
