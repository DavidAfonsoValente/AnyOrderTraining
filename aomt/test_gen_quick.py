import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

weights_path = "./weights/sft_standard-sep"
print(f"Loading model from {weights_path}...")
tokenizer = AutoTokenizer.from_pretrained(weights_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    weights_path, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="cuda:0"
)

rotary = getattr(getattr(model, "model", None), "rotary_emb", None)
if rotary is not None:
    print(f"inv_freq max: {rotary.inv_freq.abs().max().item():.4f}")
    if rotary.inv_freq.abs().max() == 0:
        print("WARNING: inv_freq is all zeros! Recomputing...")
        config = model.config
        base = config.rope_theta
        dim = int(config.hidden_size // config.num_attention_heads)
        device = rotary.inv_freq.device
        rotary.inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim)
        )
        print(f"inv_freq max after fix: {rotary.inv_freq.abs().max().item():.4f}")

model.eval()

prompt = """You are in the middle of a room. Looking quickly around you, you see a cabinet 6, a cabinet 5, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a coffeemachine 1, a countertop 3, a countertop 2, a countertop 1, a drawer 3, a drawer 2, a drawer 1, a fridge 1, a garbagecan 1, a microwave 1, a shelf 3, a shelf 2, a shelf 1, a sinkbasin 1, a stoveburner 4, a stoveburner 3, a stoveburner 2, a stoveburner 1, a toaster 1.
Your task is to: put a clean lettuce in diningtable."""

messages = [
    {"role": "system", "content": "Interact with a household to solve a task."},
    {"role": "user", "content": prompt}
]

input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
print(f"\nInput length: {input_ids.shape[1]} tokens")
print(f"Generating with iterative unmasking...")

with torch.no_grad():
    output = model.generate(
        input_ids,
        max_new_tokens=128,
        temperature=0.0,
        top_k=1,
    )

generated = output[0, input_ids.shape[1]:]
text = tokenizer.decode(generated, skip_special_tokens=True)
print(f"\n{'='*60}")
print(f"GENERATED OUTPUT:")
print(f"{'='*60}")
print(text)
print(f"{'='*60}")
