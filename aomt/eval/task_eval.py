"""
eval/task_eval.py
Full final evaluation script for ALFWorld, ScienceWorld, and WebShop.
Implements real environment interaction loops using standard benchmark APIs.
No mock results; handles real trajectories, history management, and LLaDA 2.0 generation.
"""

import argparse
import json
import os
import sys
import torch
import numpy as np
import yaml
from tqdm import tqdm
from transformers import AutoTokenizer

# Add aomt and dFactory to path for VeOmni imports
_script_dir = os.path.dirname(os.path.abspath(__file__))
_aomt_dir = os.path.dirname(_script_dir)
sys.path.insert(0, os.path.dirname(_aomt_dir))  # parent of aomt/
sys.path.insert(0, os.path.join(_aomt_dir, "dFactory"))

# Use VeOmni's model loader — handles fused MoE weights correctly
from veomni.models import build_foundation_model as _veomni_build_model
from veomni.models import build_tokenizer as _veomni_build_tokenizer

def build_tokenizer(path):
    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)

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
    env = ScienceWorldEnv("", "", envStepLimit=max_steps)
    task_names = env.get_task_names()

    scores = []
    print(f"Running ScienceWorld evaluation ({split} split, {n_episodes} episodes, {len(task_names)} tasks)...")

    ep = 0
    for task_idx, task_name in enumerate(task_names):
        if ep >= n_episodes:
            break

        try:
            env.load(task_name, 0)
            if split in ["test", "unseen"]:
                variations = env.get_variations_test()
            elif split in ["dev", "val"]:
                variations = env.get_variations_dev()
            else:
                variations = env.get_variations_train()
        except Exception as e:
            print(f"  Skipping task '{task_name}': {e}")
            continue

        if not variations:
            continue

        for var_idx in variations:
            if ep >= n_episodes:
                break

            env.load(task_name, var_idx)
            observation, info = env.reset()
            obs_history = SYSTEM_PROMPT + "\nOK\n" + str(observation)
            done = False

            for step in range(max_steps):
                if done:
                    break
                action = generate_action(model, tokenizer, obs_history, **kwargs)
                observation, reward, done, info = env.step(action)
                obs_history += f"\n{action}\n{str(observation)}"

            scores.append(info.get('score', 0.0) / 100.0)
            ep += 1

    env.close()
    return {"avg_score": float(np.mean(scores)) if scores else 0.0, "episodes": len(scores)}

def evaluate_alfworld(model, tokenizer, split="test", n_episodes=50, max_steps=50, **kwargs):
    """
    Evaluates ALFWorld and returns success rate.
    Ref: https://github.com/alfworld/alfworld
    """
    from alfworld.agents.environment import get_environment

    config_path = os.environ.get("ALFWORLD_CONFIG", "configs/alfworld_config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"ALFWorld config not found at {config_path}. Check $ALFWORLD_CONFIG.")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    split_map = {
        "test": "eval_out_of_distribution",
        "seen": "eval_in_distribution",
        "unseen": "eval_out_of_distribution",
        "train": "train",
    }
    train_eval = split_map.get(split, "eval_out_of_distribution")

    env_type = config.get("env", {}).get("type", "AlfredTWEnv")
    EnvClass = get_environment(env_type)
    tw_env = EnvClass(config, train_eval=train_eval)
    gym_env = tw_env.init_env(batch_size=1)

    print(f"Running ALFWorld evaluation ({split} -> {train_eval}, {n_episodes} episodes, {tw_env.num_games} games)...")

    successes = []
    for ep in tqdm(range(min(n_episodes, tw_env.num_games)), desc="ALFWorld episodes"):
        obs, infos = gym_env.reset()
        if isinstance(obs, (list, tuple)):
            obs = obs[0]
        obs = str(obs)

        obs_history = SYSTEM_PROMPT + "\nOK\n" + obs
        done = False
        reward = 0.0

        for step in range(max_steps):
            if done:
                break
            action = generate_action(model, tokenizer, obs_history, **kwargs)

            obs_list, reward_list, done_list, infos = gym_env.step([action])
            obs = obs_list[0] if isinstance(obs_list, (list, tuple)) else obs_list
            reward = reward_list[0] if isinstance(reward_list, (list, tuple)) else reward_list
            done = done_list[0] if isinstance(done_list, (list, tuple)) else done_list
            obs = str(obs)

            obs_history += f"\n{action}\n{obs}"

        successes.append(1.0 if reward > 0 else 0.0)

    gym_env.close()
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

def generate_action(model, tokenizer, obs_history_text: str, gen_length: int = 64, block_length: int = 32, steps: int = 256) -> str:
    """
    LLaDA 2.0 generation API.
    Utilizes bidirectional unmasking via the specialized model.generate call.

    Training format: a single user message containing the full concatenated
    history (system prompt, OK, observations, actions joined by \\n), with the
    assistant response being the next action.  We replicate that here.
    """
    messages = [
        {"role": "user", "content": obs_history_text},
    ]

    token_output = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    )
    if isinstance(token_output, torch.Tensor):
        input_ids = token_output
    elif hasattr(token_output, "input_ids"):
        input_ids = token_output.input_ids
    else:
        input_ids = torch.tensor([token_output], dtype=torch.long)

    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    target_device = next(model.parameters()).device
    input_ids = input_ids.to(target_device)

    max_context = getattr(model.config, "max_position_embeddings", 2048) - gen_length - 8
    if input_ids.shape[1] > max_context:
        input_ids = input_ids[:, -max_context:]

    _debug_counter = getattr(generate_action, '_call_count', 0)
    generate_action._call_count = _debug_counter + 1
    _do_log = (_debug_counter < 5) or (_debug_counter % 50 == 0)

    if _do_log:
        print(f"\n[DEBUG gen #{_debug_counter}] input_ids shape: {input_ids.shape}, "
              f"n_messages: {len(messages)}, mask_id: {tokenizer.mask_token_id or 156895}")

    with torch.no_grad():
        output_ids = model.generate(
            inputs=input_ids,
            gen_length=gen_length,
            block_length=block_length,
            steps=steps,
            temperature=0.2,
            eos_early_stop=True,
            mask_id=tokenizer.mask_token_id or 156895
        )

    generated = output_ids[0, input_ids.shape[1]:]

    if _do_log:
        raw_tokens = generated[:60].tolist()
        mask_count = (generated == (tokenizer.mask_token_id or 156895)).sum().item()
        eos_count = (generated == tokenizer.eos_token_id).sum().item() if tokenizer.eos_token_id else 0
        print(f"[DEBUG gen #{_debug_counter}] generated length: {generated.shape[0]}, "
              f"mask_tokens_remaining: {mask_count}, eos_tokens: {eos_count}")
        print(f"[DEBUG gen #{_debug_counter}] first 60 token ids: {raw_tokens}")

    eos_id = tokenizer.eos_token_id
    if eos_id is not None:
        eos_positions = (generated == eos_id).nonzero(as_tuple=True)[0]
        if len(eos_positions) > 0:
            generated = generated[:eos_positions[0]]

    full_response = tokenizer.decode(generated, skip_special_tokens=True).strip()

    if _do_log:
        print(f"[DEBUG gen #{_debug_counter}] full_response (first 300 chars): {repr(full_response[:300])}")

    # ── Clean MDLM artifacts: duplicated role tokens ──
    import re
    # "ActionActionAction:" → "Action:", "ThoughtThought:" → "Thought:" etc.
    cleaned = re.sub(r'(Action){2,}', 'Action', full_response)
    cleaned = re.sub(r'(Thought){2,}', 'Thought', cleaned)
    cleaned = re.sub(r'(Observation){2,}', 'Observation', cleaned)

    # ── Parse action ──
    action = ""
    if "Action:" in cleaned:
        # Take the FIRST Action: match, truncate at newline
        action_part = cleaned.split("Action:")[1].strip()
        action = action_part.split("\n")[0].strip()
    elif "action:" in cleaned:
        action_part = cleaned.split("action:")[1].strip()
        action = action_part.split("\n")[0].strip()
    
    # If no action found or action is empty, try to find a valid ALFWorld verb pattern
    if not action or len(action) < 3:
        verb_pattern = re.search(
            r'\b(go to|take|put|open|close|toggle|clean|heat|cool|examine|look|use|inventory)\b.+',
            cleaned, re.IGNORECASE
        )
        if verb_pattern:
            action = verb_pattern.group(0).split("\n")[0].strip()
        else:
            # Last resort: take last non-empty line
            lines = [l.strip() for l in cleaned.split("\n") if l.strip()]
            action = lines[-1] if lines else cleaned

    # ── Final cleanup ──
    action = action.rstrip('.').strip()
    # Remove stray role prefixes that slipped through
    action = re.sub(r'^(Thought|Action|Observation)\s*:?\s*', '', action).strip()
    # Truncate unreasonably long actions (valid ALFWorld actions are <10 words)
    words = action.split()
    if len(words) > 10:
        action = ' '.join(words[:10])

    if _do_log:
        print(f"[DEBUG gen #{_debug_counter}] parsed action: {repr(action[:200])}")

    return action

# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="AOMT Task Evaluation Suite")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--config_path", type=str, default="./weights/llada2-mini-merged",
                        help="Path to base model (for config/architecture). Default: ./weights/llada2-mini-merged")
    parser.add_argument("--tokenizer", type=str, required=True, help="Path to the tokenizer")
    parser.add_argument("--benchmark", type=str, required=True, choices=["alfworld", "scienceworld", "webshop"], 
                        help="Benchmark to evaluate")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test", "seen", "unseen", "dev", "val"], 
                        help="Data split to evaluate on")
    parser.add_argument("--n_episodes", type=int, default=50, help="Number of episodes to evaluate")
    parser.add_argument("--gen_length", type=int, default=64, help="Max length of generated action")
    parser.add_argument("--block_length", type=int, default=32, help="Block length for LLaDA generation")
    parser.add_argument("--steps", type=int, default=256, help="Denoising steps per block")
    parser.add_argument("--output_file", type=str, required=True, help="JSON file to save results")
    parser.add_argument("--device", type=str, default=None, help="Device to load model on, e.g. 'cuda:0'. Default: auto")
    args = parser.parse_args()

    init_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from {args.model_path} (config from {args.config_path})...")

    tokenizer = build_tokenizer(args.tokenizer)
    model = _veomni_build_model(
        weights_path=args.model_path,
        config_path=args.config_path,
        torch_dtype="bfloat16",
        attn_implementation="sdpa",
        init_device=init_device,
        moe_implementation="fused"
    )
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
