# aomt/inference.py

import torch
import inspect
from transformers import AutoModelForCausalLM, AutoTokenizer
from aomt.utils.tokenizer_utils import get_mask_token_id, extract_action_from_react


def load_model_for_eval(model_path: str, tokenizer_path: str):
    """
    Load fine-tuned model in SEP format for inference.
    SEP format = standard HuggingFace MoE checkpoint.
    MERGED format is for training only — never use it here.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    MASK_TOKEN_ID = get_mask_token_id(tokenizer_path)  # verifies == 156895

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()

    # Probe whether eos_early_stop is supported
    model._supports_eos_early_stop = (
        "eos_early_stop" in inspect.signature(model.generate).parameters
    )

    return model, tokenizer, MASK_TOKEN_ID


def generate_action(
    model,
    tokenizer,
    conversation: list,      # [{"role":"user","content":…}, {"role":"assistant",…}, …]
    gen_length: int = 256,
    steps: int = 32,         # denoising steps; use 1 for one-shot
    temperature: float = 0.0,
) -> str:
    """
    Generate one assistant turn using LLaDA2.0-mini masked diffusion.
    Uses chat template — correct for Standard SFT and Prefix SFT.
    """
    # For one-shot, use a single block of size gen_length
    block_length = gen_length if steps == 1 else 32
    assert gen_length % block_length == 0, \
        f"gen_length={gen_length} must be divisible by block_length={block_length}"

    input_ids = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    ).to(next(model.parameters()).device)

    prompt_len = input_ids.shape[1]

    generate_kwargs = dict(
        gen_length=gen_length,
        block_length=block_length,
        steps=steps,
        temperature=temperature,
        cfg_scale=0.0,
        remasking="low_confidence",
    )
    if model._supports_eos_early_stop:
        generate_kwargs["eos_early_stop"] = True

    with torch.no_grad():
        output_ids = model.generate(input_ids, **generate_kwargs)

    generated = output_ids[0, prompt_len:]

    # EOS truncation
    eos_id = tokenizer.eos_token_id
    eos_positions = (generated == eos_id).nonzero(as_tuple=True)[0]
    if len(eos_positions) > 0:
        generated = generated[:eos_positions[0]]

    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def generate_action_flat(
    model,
    tokenizer,
    history_parts: list,     # list of strings (obs/act alternating)
    gen_length: int = 256,
    steps: int = 32,
    temperature: float = 0.0,
) -> str:
    """
    Generate one assistant turn using AOMT-Mixed flat trajectory format.
    Matches Training: O_0 [EOS] A_0 [EOS] ... O_t [EOS] [MASK]*gen_length
    """
    sep_id = tokenizer.eos_token_id
    all_ids = []
    for i, text in enumerate(history_parts):
        ids = tokenizer.encode(text, add_special_tokens=False)
        all_ids.extend(ids)
        # Trailing EOS after every element
        all_ids.append(sep_id)

    input_ids = torch.tensor([all_ids], dtype=torch.long).to(next(model.parameters()).device)
    prompt_len = input_ids.shape[1]

    block_length = 32
    if steps == 1: block_length = gen_length
    
    assert gen_length % block_length == 0

    generate_kwargs = dict(
        gen_length=gen_length,
        block_length=block_length,
        steps=steps,
        temperature=temperature,
        cfg_scale=0.0,
        remasking="low_confidence",
    )
    if model._supports_eos_early_stop:
        generate_kwargs["eos_early_stop"] = True

    with torch.no_grad():
        output_ids = model.generate(input_ids, **generate_kwargs)

    generated = output_ids[0, prompt_len:]

    # EOS truncation
    eos_positions = (generated == sep_id).nonzero(as_tuple=True)[0]
    if len(eos_positions) > 0:
        generated = generated[:eos_positions[0]]

    return tokenizer.decode(generated, skip_special_tokens=True).strip()

