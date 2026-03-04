# aomt/eval/nll_obs.py
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from aomt.data.unit_parser import TokenizedTrajectory
from transformers import AutoModelForCausalLM, AutoTokenizer

@torch.no_grad()
def compute_nll_obs(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    tokenized_trajectories: list[TokenizedTrajectory],
    batch_size: int = 4,
    device: str = "cuda"
) -> dict:
    """
    Computes the Observation-Masked Negative Log-Likelihood (NLL-obs).

    This metric evaluates the model's implicit world model by measuring how well
    it can reconstruct a single observation (O_t) given all other units in the
    trajectory as bidirectional context.

    For each observation unit O_t in each trajectory:
      1. Create a copy of the trajectory's tokens.
      2. Replace all tokens in the O_t span with the MASK token.
      3. Perform a full bidirectional forward pass with this masked input.
      4. Calculate the cross-entropy loss for the tokens in the O_t span against
         the original, unmasked tokens.
      5. Average this NLL across all observations to get the final score.

    Args:
        model: The trained model to evaluate (must support bidirectional attention).
        tokenizer: The tokenizer used for training.
        tokenized_trajectories: A list of tokenized trajectories for the eval set.
        batch_size (int): The batch size for evaluation.
        device (str): The device to run evaluation on ("cuda" or "cpu").

    Returns:
        dict: A dictionary containing the mean NLL-obs, and breakdowns by
              environment and by step position.
    """
    model.eval()
    model.to(device)
    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        raise ValueError("Tokenizer must have a `mask_token_id`.")

    # Store NLL values for aggregation
    results = {
        "all_nll": [],
        "nll_by_env": defaultdict(list),
        "nll_by_position": defaultdict(list),
    }

    # Create batches of inputs to evaluate
    eval_inputs = []
    for traj_idx, traj in enumerate(tokenized_trajectories):
        obs_units = [u for u in traj.unit_spans if u.unit_type == "obs"]
        for obs_unit in obs_units:
            if obs_unit.token_start >= obs_unit.token_end:
                continue # Skip empty units

            masked_ids = traj.input_ids.clone()
            masked_ids[obs_unit.token_start:obs_unit.token_end] = mask_token_id
            
            eval_inputs.append({
                "masked_ids": masked_ids,
                "target_ids": traj.input_ids,
                "target_span": slice(obs_unit.token_start, obs_unit.token_end),
                "env": traj.env,
                "step": obs_unit.unit_index // 2,
            })

    print(f"Starting NLL-obs evaluation on {len(eval_inputs)} observation units...")
    for i in tqdm(range(0, len(eval_inputs), batch_size), desc="Evaluating NLL-obs"):
        batch_data = eval_inputs[i:i+batch_size]
        
        # Prepare batch tensors
        max_len = max(len(d["masked_ids"]) for d in batch_data)
        input_ids_batch = []
        attention_mask_batch = []
        
        for data in batch_data:
            padding_len = max_len - len(data["masked_ids"])
            input_ids_batch.append(F.pad(data["masked_ids"], (0, padding_len), value=tokenizer.pad_token_id))
            attention_mask_batch.append(F.pad(torch.ones_like(data["masked_ids"]), (0, padding_len), value=0))

        input_ids_tensor = torch.stack(input_ids_batch).to(device)
        attention_mask_tensor = torch.stack(attention_mask_batch).to(device)

        # Forward pass with bidirectional attention
        logits = model(
            input_ids=input_ids_tensor,
            attention_mask=attention_mask_tensor,
        ).logits

        # Calculate NLL for each item in the batch
        for j, data in enumerate(batch_data):
            target_span = data["target_span"]
            
            # Extract logits and targets for the specific observation span
            obs_logits = logits[j, target_span]
            obs_target = data["target_ids"][target_span].to(device)

            if obs_logits.shape[0] == 0:
                continue

            nll = F.cross_entropy(obs_logits, obs_target, reduction="mean").item()

            # Store results
            results["all_nll"].append(nll)
            results["nll_by_env"][data["env"]].append(nll)
            results["nll_by_position"][data["step"]].append(nll)

    # Aggregate and return final metrics
    final_metrics = {
        "mean_nll_obs": float(np.mean(results["all_nll"])) if results["all_nll"] else 0.0,
        "nll_by_env": {
            k: float(np.mean(v)) for k, v in results["nll_by_env"].items()
        },
        "nll_by_position": {
            k: float(np.mean(v)) for k, v in sorted(results["nll_by_position"].items())
        },
    }

    return final_metrics
