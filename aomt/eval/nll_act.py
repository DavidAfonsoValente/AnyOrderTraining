# aomt/eval/nll_act.py
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from aomt.data.unit_parser import TokenizedTrajectory
from transformers import AutoModelForCausalLM, AutoTokenizer

@torch.no_grad()
def compute_nll_act(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    tokenized_trajectories: list[TokenizedTrajectory],
    batch_size: int = 4,
    device: str = "cuda"
) -> dict:
    """
    Computes the Action-Masked Negative Log-Likelihood (NLL-act).

    This is a diagnostic metric, analogous to NLL-obs, but for actions. It
    measures how well the model can reconstruct a single action (A_t) given
    all other units in the trajectory as bidirectional context.

    This metric is computable for any model that can use bidirectional attention
    (i.e., AOMT models). It helps diagnose policy learning.

    Args:
        model: The trained model to evaluate.
        tokenizer: The tokenizer used for training.
        tokenized_trajectories: A list of tokenized trajectories for the eval set.
        batch_size (int): The batch size for evaluation.
        device (str): The device to run evaluation on ("cuda" or "cpu").

    Returns:
        dict: A dictionary containing the mean NLL-act, and breakdowns by
              environment and by step position.
    """
    model.eval()
    model.to(device)
    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        raise ValueError("Tokenizer must have a `mask_token_id`.")

    results = {
        "all_nll": [],
        "nll_by_env": defaultdict(list),
        "nll_by_position": defaultdict(list),
    }

    eval_inputs = []
    for traj in tokenized_trajectories:
        act_units = [u for u in traj.unit_spans if u.unit_type == "act"]
        for act_unit in act_units:
            if act_unit.token_start >= act_unit.token_end:
                continue

            masked_ids = traj.input_ids.clone()
            masked_ids[act_unit.token_start:act_unit.token_end] = mask_token_id
            
            eval_inputs.append({
                "masked_ids": masked_ids,
                "target_ids": traj.input_ids,
                "target_span": slice(act_unit.token_start, act_unit.token_end),
                "env": traj.env,
                "step": act_unit.unit_index // 2,
            })

    print(f"Starting NLL-act evaluation on {len(eval_inputs)} action units...")
    for i in tqdm(range(0, len(eval_inputs), batch_size), desc="Evaluating NLL-act"):
        batch_data = eval_inputs[i:i+batch_size]
        
        max_len = max(len(d["masked_ids"]) for d in batch_data)
        input_ids_batch = []
        attention_mask_batch = []
        
        for data in batch_data:
            padding_len = max_len - len(data["masked_ids"])
            input_ids_batch.append(F.pad(data["masked_ids"], (0, padding_len), value=tokenizer.pad_token_id))
            attention_mask_batch.append(F.pad(torch.ones_like(data["masked_ids"]), (0, padding_len), value=0))

        input_ids_tensor = torch.stack(input_ids_batch).to(device)
        attention_mask_tensor = torch.stack(attention_mask_batch).to(device)

        logits = model(
            input_ids=input_ids_tensor,
            attention_mask=attention_mask_tensor,
        ).logits

        for j, data in enumerate(batch_data):
            target_span = data["target_span"]
            act_logits = logits[j, target_span]
            act_target = data["target_ids"][target_span].to(device)

            if act_logits.shape[0] == 0:
                continue

            nll = F.cross_entropy(act_logits, act_target, reduction="mean").item()

            results["all_nll"].append(nll)
            results["nll_by_env"][data["env"]].append(nll)
            results["nll_by_position"][data["step"]].append(nll)

    final_metrics = {
        "mean_nll_act": float(np.mean(results["all_nll"])) if results["all_nll"] else 0.0,
        "nll_by_env": {
            k: float(np.mean(v)) for k, v in results["nll_by_env"].items()
        },
        "nll_by_position": {
            k: float(np.mean(v)) for k, v in sorted(results["nll_by_position"].items())
        },
    }

    return final_metrics
