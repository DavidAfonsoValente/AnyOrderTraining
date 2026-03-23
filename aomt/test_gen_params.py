import sys, torch, json, os
sys.stdout.reconfigure(line_buffering=True)
os.environ["VEOMNI_USE_LIGER_KERNEL"] = "0"
sys.path.insert(0, 'dFactory/VeOmni')
sys.path.insert(0, 'dFactory')
sys.path.insert(0, 'eval')
from task_eval import build_tokenizer, build_foundation_model

tok = build_tokenizer('./weights/LLaDA2.0-mini')
print("Loading model...")
model = build_foundation_model('./weights/sft_standard-sep', init_device='cuda:0')
model.eval()

# A simple, realistic ALFWorld prompt
test_prompt = """Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. 
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
Action: <your next action>
OK
You are in the middle of a room. Looking quickly around you, you see a bed 1, a desk 2, a desk 1, a drawer 6, a drawer 5, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a garbagecan 1, a laundryhamper 1, a safe 1, a shelf 6, a shelf 5, a shelf 4, a shelf 3, a shelf 2, and a shelf 1.
Your task is to: examine the pen with the desklamp."""

messages = [{"role": "user", "content": test_prompt}]
token_output = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt")
if isinstance(token_output, torch.Tensor):
    input_ids = token_output
elif hasattr(token_output, "input_ids"):
    input_ids = token_output.input_ids
else:
    input_ids = torch.tensor([token_output], dtype=torch.long)

if input_ids.dim() == 1:
    input_ids = input_ids.unsqueeze(0)

input_ids = input_ids.to('cuda:0')

# Hyperparameter combinations to test
configs = [
    {"steps": 16, "block_length": 32, "temperature": 0.0, "top_k": 1},      # Current eval settings
    {"steps": 64, "block_length": 32, "temperature": 0.0, "top_k": 1},      # More steps, greedy
    {"steps": 128, "block_length": 32, "temperature": 0.0, "top_k": 1},     # Even more steps, greedy
    {"steps": 32, "block_length": 16, "temperature": 0.0, "top_k": 1},      # Smaller blocks
    {"steps": 64, "block_length": 32, "temperature": 0.7, "top_k": 50},     # Sampling (what base model uses)
    {"steps": 128, "block_length": 64, "temperature": 0.7, "top_k": 50},    # LLaDA paper default-ish
]

print(f"\nTesting generation with {len(configs)} configurations...")
print("="*80)

for i, cfg in enumerate(configs):
    print(f"Config {i+1}: steps={cfg['steps']}, block_length={cfg['block_length']}, temp={cfg['temperature']}, top_k={cfg['top_k']}")
    
    with torch.no_grad():
        output_ids = model.generate(
            inputs=input_ids,
            gen_length=64,  # Keep it short for testing
            block_length=cfg['block_length'],
            steps=cfg['steps'],
            temperature=cfg['temperature'],
            top_k=cfg['top_k'],
            eos_early_stop=True,
            mask_id=tok.mask_token_id or 156895
        )
        
    generated = output_ids[0, input_ids.shape[1]:]
    eos_id = tok.eos_token_id
    if eos_id is not None:
        eos_positions = (generated == eos_id).nonzero(as_tuple=True)[0]
        if len(eos_positions) > 0:
            generated = generated[:eos_positions[0]]
            
    full_response = tok.decode(generated, skip_special_tokens=True).strip()
    print(f"Output: {repr(full_response)}")
    print("-" * 80)

print("Done.")
