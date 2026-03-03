import argparse
import logging
import os
import sys
import json
from collections import defaultdict
from typing import List, Dict, Any, Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer
from alfworld.agents.environment.alfworld_env import AlfworldEnv
from alfworld.info import ALFWORLD_DATA

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

def format_alfworld_obs(observation: str, goal: str = "") -> str:
    """Formats an ALFWorld observation for a causal LM input."""
    # This format can be refined based on what works best for the Qwen model.
    if goal:
        return f"Goal: {goal}
Observation: {observation}
Action:"
    return f"Observation: {observation}
Action:"

def parse_action_from_model_output(model_output: str) -> str:
    """
    Parses the generated text from the model to extract a valid ALFWorld action.
    This is a crucial and potentially complex step. The model might generate
    more than just the action.
    """
    # Simple heuristic: take the first line as the action.
    # More robust parsing (e.g., regex, looking for specific keywords) might be needed.
    action = model_output.strip().split('
')[0].strip()
    return action

def run_alfworld_episode(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    env: AlfworldEnv,
    max_steps: int = 50,
    temperature: float = 0.7,
    top_p: float = 0.9,
    episode_log: List[Dict[str, Any]] = None,
) -> Tuple[bool, List[Dict[str, Any]]]:
    """Runs a single episode in ALFWorld with the given model."""
    if episode_log is None:
        episode_log = []

    ob, info = env.reset() # ob is observation, info contains initial goal
    current_obs = ob[0] # Take the first observation as the primary one
    goal = info["extra.game_object_id"] # Example of extracting goal from info

    trajectory_history = []
    success = False

    for step in range(max_steps):
        formatted_input = format_alfworld_obs(current_obs, goal)
        trajectory_history.append({"observation": current_obs})

        input_ids = tokenizer.encode(formatted_input, return_tensors="pt").to(model.device)
        
        # Generate next token, then decode and parse for action
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=20, # Generate up to 20 new tokens for the action
                temperature=temperature,
                do_sample=True,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id, # Or use tokenizer.pad_token_id if defined
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode the generated part, excluding the input prompt
        generated_text = tokenizer.decode(output[0, input_ids.shape[1]:], skip_special_tokens=True)
        action_text = parse_action_from_model_output(generated_text)
        
        trajectory_history.append({"action": action_text})

        # Step in the environment
        try:
            ob, reward, done, info = env.step([action_text])
            current_obs = ob[0]
            if done:
                success = bool(reward) # Assuming reward is 1 for success, 0 for failure
                break
        except Exception as e:
            logger.warning(f"Invalid action '{action_text}' or environment error: {e}")
            done = True # Consider invalid action as ending the episode
            success = False
            break
    
    # Add final observation
    if not done: # If max_steps reached without done=True
        trajectory_history.append({"observation": current_obs})
    elif "observation" not in trajectory_history[-1]: # If episode ended, add final obs if not already there
        trajectory_history.append({"observation": current_obs})

    return success, trajectory_history


def main():
    parser = argparse.ArgumentParser(description="Generate rollouts from a Qwen model in ALFWorld.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned Qwen model (IL).")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the generated rollout trajectories (JSONL).")
    parser.add_argument("--num_episodes", type=int, default=100, help="Number of ALFWorld episodes to run for rollouts.")
    parser.add_argument("--max_steps_per_episode", type=int, default=50, help="Maximum steps per episode in ALFWorld.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature for text generation.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter for text generation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    
    args = parser.parse_args()

    # Setup logging
    logger.info(f"Arguments: {args}")
    
    # Set seed
    torch.manual_seed(args.seed)
    # If running on GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Load tokenizer and model
    logger.info(f"Loading model and tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True)
    model.eval() # Set model to evaluation mode
    
    if torch.cuda.is_available():
        model.to("cuda")
        logger.info("Model moved to GPU.")
    else:
        logger.info("Running model on CPU.")

    # Initialize ALFWorld environment
    # The ALFWorld data path is typically managed by alfworld-download
    alfworld_data_path = os.path.expanduser(ALFWORLD_DATA) # ALFWORLD_DATA is ~/.cache/alfworld
    logger.info(f"ALFWorld data path: {alfworld_data_path}")
    
    # It's crucial that `alfworld-download --extra` has been run and the data is present.
    # The environment needs to be created with the correct split and setup.
    # We will use 'eval_out_of_distribution' as a common evaluation split.
    # If you need to specify a different split or config, adapt this.
    env = AlfworldEnv(
        alfworld_data_path=alfworld_data_path,
        split="eval_out_of_distribution",
        mode="json" # Interact with JSON-formatted games
    )
    logger.info("ALFWorld environment initialized.")

    all_rollout_trajectories = []
    
    for i in range(args.num_episodes):
        logger.info(f"Running episode {i+1}/{args.num_episodes}...")
        success, trajectory = run_alfworld_episode(model, tokenizer, env, args.max_steps_per_episode, args.temperature, args.top_p)
        
        all_rollout_trajectories.append({
            "episode_id": i,
            "success": success,
            "trajectory": trajectory
        })
        logger.info(f"Episode {i+1} finished. Success: {success}")

    # Save all generated rollouts to a JSONL file
    with open(args.output_file, 'w') as f:
        for entry in all_rollout_trajectories:
            json.dump(entry, f)
            f.write('
')
    
    logger.info(f"All {len(all_rollout_trajectories)} rollout trajectories saved to {args.output_file}")
    logger.info("Rollout generation complete.")

if __name__ == "__main__":
    main()
