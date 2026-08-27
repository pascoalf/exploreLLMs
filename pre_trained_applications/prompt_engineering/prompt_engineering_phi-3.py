"""from https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch06.html#in_context_learning_providing_examples"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map = "cuda",
    torch_dtype = "auto",
    trust_remote_code = True
)
#
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

# create a pipeline
pipe = pipeline(
    "text-generation",
    model = model,
    tokenizer=tokenizer,
    return_full_text = False,
    max_new_tokens = 500,
    do_sample = False
)

# Prompt
messages = [
    {"role": "user", "content": """
    The sky is... 
    
    [select a color]
    """}
]

# Generate the output
output = pipe(messages)
print(output[0]["generated_text"])

# To access the underlying tokenizer
prompt = pipe.tokenizer.apply_chat_template(messages, tokenize = False)
print(prompt)

# To introduce more creativity, increase the temperature
ht_output = pipe(messages, do_sample=True, temperature=1)
print(ht_output[0]["generated_text"])

# Using a high top_p
htp_output = pipe(messages, do_sample=True, top_p=0.5, temperature = 0.2)
print(htp_output[0]["generated_text"])

# More complex prompt engineering
persona = "You are a painter.\n"
instruction = "Suggest a better color for the sky.\n"
context = "You are in front of a blank canvas.\n"
data_format = " Justify.\n"
audience = "(you are talking to yourself)\n"
tone = "The tone is excentric.\n"
#text = "MY TEXT TO SUMMARIZE"
#data = f"Text to summarize: {text}"

# The full prompt - remove and add pieces to view its impact on the generated output
query = persona + instruction + context + data_format + audience + tone #+ data

# compare
# Using a high top_p + low temp
query1 = pipe(query, do_sample=True, top_p=0.8, temperature = 0.1)
print(query1[0]["generated_text"])

# high temp and high top_p (most creative)
query2 = pipe(query, do_sample=True, top_p=1, temperature = 1)
print(query2[0]["generated_text"])