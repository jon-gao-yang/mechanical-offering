# Sources:
# https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct?library=transformers
# https://huggingface.co/docs/transformers/en/conversations

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os

path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs/Llama-1B-GRPO/checkpoint-7473')
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path)
pipeline = pipeline(task="text-generation", tokenizer=tokenizer, model=model, device_map="auto")
chat = [{"role": "user", "content": "Hi."},]

print("\nCONVERSATION BEGINS (type 'exit()' to quit)")
while True:
	response = pipeline(chat, max_new_tokens=786)
	print(response[0]["generated_text"][-1]["content"])
	prompt = input("\n>>> ")

	if prompt == "exit()": break
	chat = response[0]["generated_text"]
	chat.append({"role": "user", "content": prompt})