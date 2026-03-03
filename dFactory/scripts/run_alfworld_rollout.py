import os
import json
import argparse
from tqdm import tqdm
import torch
import alfworld.agents.environment as environment
from transformers import AutoTokenizer
from veomni.models import build_model

# A simple function to clean the model's output
def clean_action(action_text):
    # Actions are typically single-line commands
    return action_text.split('
')[0].strip()

def run_rollouts(model_path, num_episodes, output_file):
    """
    Runs rollouts in the ALFWorld environment to evaluate a trained model.
    """
    # 1. Load Model and Tokenizer
    print(f"Loading model and tokenizer from: {model_path}")
    
    # Use bfloat16 for inference as specified in the base configs
    dtype = torch.bfloat16
    
    try:
        # The VeOmni way seems to be using build_model
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = build_model(model_path, torch_dtype=dtype, trust_remote_code=True)
        model.cuda()
        model.eval()
        print("Model loaded successfully via build_model.")
    except Exception as e:
        print(f"Failed to load model with build_model: {e}")
        print("Falling back to standard Hugging Face AutoModelForCausalLM.")
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
        model.cuda()
        model.eval()

    # 2. Setup ALFWorld Environment
    # This config requires the `tw_games` directory to be in the right place.
    # Users should follow ALFWorld's setup instructions.
    env_config = {
        'name': 'alfworld',
        'use_history': True,
        'use_memory': True,
        'train_eval': 'eval_out_of_distribution' # As per standard ALFWorld eval
    }
    env = environment.StatefulAlfworld(env_config)
    print("ALFWorld environment created.")

    successes = 0
    
    # 3. Run Episode Loop
    for i in tqdm(range(num_episodes), desc="Running ALFWorld Episodes"):
        obs, info = env.reset()
        done = False
        
        # The initial observation from the env is a list of strings
        current_history = " ".join(obs)
        
        pbar = tqdm(total=50, desc=f"Episode {i+1}", leave=False) # Max 50 steps per episode
        step_count = 0
        while not done and step_count < 50:
            # Format the history for the model. The prompt should encourage an action.
            prompt = f"{current_history}
> "
            
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=50, # Actions are short
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    do_sample=False # Use deterministic generation for eval
                )
            
            generated_text = tokenizer.decode(generated_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            action = clean_action(generated_text)
            
            # Interact with environment
            obs, reward, done, info = env.step([action])
            
            # Update history
            current_history += f"
> {action}
{obs}"
            step_count += 1
            pbar.update(1)

        pbar.close()

        # Check if the episode was successful by looking for the success message in the last obs
        if "You successfully" in obs:
            successes += 1
            print(f"Episode {i+1}: Success")
        else:
            print(f"Episode {i+1}: Failure")
            
    # 4. Calculate and Save Results
    success_rate = successes / num_episodes
    print(f"
{'='*20} Evaluation Complete {'='*20}")
    print(f"Total Episodes: {num_episodes}")
    print(f"Successes: {successes}")
    print(f"Success Rate: {success_rate:.4f}")
    print(f"{'='*59}")

    results = {
        "model_path": model_path,
        "num_episodes": num_episodes,
        "successes": successes,
        "success_rate": success_rate,
    }
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ALFWorld rollouts to evaluate a model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model checkpoint (hf_ckpt directory).")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the output JSON results.")
    parser.add_argument("--num_episodes", type=int, default=100, help="Number of episodes to run for evaluation.")
    args = parser.parse_args()
    
    run_rollouts(args.model_path, args.num_episodes, args.output_file)
