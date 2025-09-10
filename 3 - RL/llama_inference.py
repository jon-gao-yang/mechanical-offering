# from https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct?library=transformers

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# original
path = "meta-llama/Llama-3.2-1B-Instruct"

# after running train_grpo.py
path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs/Llama-1B-GRPO/checkpoint-7473')

tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path)

# original
messages = [{"role": "user", "content": "Who are you?"},]

# after running train_grpo.py
messages = [{"role": "user", "content": "If Bob has 10 apples and gives half to his friend Sally, how many apple pies can Sally make?"},]

inputs = tokenizer.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))