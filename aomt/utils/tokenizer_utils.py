# aomt/utils/tokenizer_utils.py

from transformers import AutoTokenizer

_TOK_CACHE: dict = {}

def get_tokenizer(path: str):
    if path not in _TOK_CACHE:
        _TOK_CACHE[path] = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    return _TOK_CACHE[path]

def get_mask_token_id(tokenizer_path: str) -> int:
    tok = get_tokenizer(tokenizer_path)
    mid = tok.mask_token_id
    assert mid is not None, f"mask_token_id is None for {tokenizer_path}"
    # LLaDA2.0-mini mask_token_id is 156895. LLaDA 8B is 126336.
    assert mid == 156895, f"Unexpected mask_token_id={mid} (expected 156895 for LLaDA2.0-mini)"
    return mid

def extract_action_from_react(text: str) -> str:
    """Extract ALFWorld/ScienceWorld action from ReAct-format generation."""
    if "Action:" in text:
        after = text.split("Action:")[-1].strip()
        return after.split("\n")[0].strip()
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    return lines[-1] if lines else text.strip()
