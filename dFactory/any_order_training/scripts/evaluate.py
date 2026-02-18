
import argparse
import json
import torch
from torch.nn.functional import cross_entropy
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

def calculate_nll(model, tokenizer, trajectory, device):
    """
    Calculates the Negative Log-Likelihood of a trajectory.
    """
    # The trajectory is a list of messages. First, apply the chat template
    inputs_str = tokenizer.apply_chat_template(trajectory, tokenize=False, add_generation_prompt=False)
    
    # Tokenize the string
    tokenized_input = tokenizer(
        inputs_str,
        return_tensors="pt",
        truncation=True,
        max_length=model.config.max_position_embeddings,
        add_special_tokens=False
    )
    
    input_ids = tokenized_input.input_ids.to(device)
    attention_mask = tokenized_input.attention_mask.to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    # Shift logits and labels for next-token prediction
    shifted_logits = logits[..., :-1, :].contiguous()
    labels = input_ids[..., 1:].contiguous()
    
    # Calculate loss
    loss = cross_entropy(shifted_logits.view(-1, shifted_logits.size(-1)), labels.view(-1), reduction='mean')
    
    return loss.item()

def evaluate_task_success_rate(model, tokenizer, env_name):
    """
    Placeholder for evaluating the task success rate.
    
    This function requires running the agent in the environment, which is not
    possible in the current setup without the required libraries.
    """
    print("="*50)
    print("WARNING: Task success rate evaluation is not implemented.")
    print("This requires the `gymnasium` and `minigrid` libraries to be installed,")
    print("and an interactive agent loop to be implemented.")
    print("="*50)
    return "Not Implemented"


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model.")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the tokenizer.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the test dataset (JSONL).")
    parser.add_argument("--env_name", type=str, default="BabyAI-GoToRedBall-v0", help="Name of the environment for task success rate evaluation.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(args.model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model.eval()

    # --- NLL Evaluation ---
    total_nll = 0
    num_trajectories = 0
    print(f"Evaluating NLL on {args.dataset_path}...")
    with open(args.dataset_path, 'r') as f:
        for line in tqdm(f):
            data = json.loads(line)
            trajectory = data["messages"]
            nll = calculate_nll(model, tokenizer, trajectory, device)
            total_nll += nll
            num_trajectories += 1
    
    avg_nll = total_nll / num_trajectories if num_trajectories > 0 else 0
    print(f"
Average NLL: {avg_nll:.4f}")

    # --- Task Success Rate Evaluation ---
    success_rate = evaluate_task_success_rate(model, tokenizer, args.env_name)
    print(f"Task Success Rate: {success_rate}")


if __name__ == "__main__":
    main()
