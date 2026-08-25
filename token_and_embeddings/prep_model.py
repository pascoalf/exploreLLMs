"""This script is just following the example code in chapter 3 with small modifications
https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch03.html 
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
)

# Create a pipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    max_new_tokens=50,
    do_sample=False,
)

prompt = "Write an Hello world."

output = generator(prompt)

print(output[0]['generated_text'])

# Inspect LM head
prompt = "The capital of France is"

# Tokenize the input prompt
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

# Tokenize the input prompt
input_ids = input_ids.to("cuda")

# Get the output of the model before the lm_head
model_output = model.model(input_ids)

# Get the output of the lm_head
lm_head_output = model.lm_head(model_output[0])

print(lm_head_output)

token_id = lm_head_output[0,-1].argmax(-1)

print(token_id)

print(tokenizer.decode(token_id))

# dimension of the matrices
print(model_output[0].shape)

# output oof LM head
print(lm_head_output.shape)

