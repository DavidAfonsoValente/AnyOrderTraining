from transformers import AutoTokenizer
import os

model_path = "/home/davidvalente/AnyOrderTraining/aomt/weights/LLaDA2.0-mini"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

print(f"BOS: {tokenizer.bos_token} (id: {tokenizer.bos_token_id})")
print(f"EOS: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")
print(f"PAD: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
print(f"MASK: {tokenizer.mask_token} (id: {tokenizer.mask_token_id})")

# Check if 156895 is indeed MASK
try:
    decoded = tokenizer.decode([156895])
    print(f"Token 156895: '{decoded}'")
except Exception as e:
    print(f"Error decoding 156895: {e}")

# Check chat template
try:
    conversation = [{"role": "user", "content": "Hello, how are you?"}]
    template_output = tokenizer.apply_chat_template(conversation, tokenize=False)
    print(f"Chat template output: \n{template_output}")
except Exception as e:
    print(f"Error with chat template: {e}")
