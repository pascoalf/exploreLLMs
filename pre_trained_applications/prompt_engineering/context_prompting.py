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


# one_shot_prompt, i.e., one single example
one_shot_prompt = [
 {
        "role": "user",
        "content": "A 'Gigamuru' is a type of Japanese musical instrument. An example of a sentence that uses the word Gigamuru is:"
    },
    {
        "role": "assistant",
        "content": "I have a Gigamuru that my uncle gave me as a gift. I love to play it at home."
    },
    {
        "role": "user",
        "content": "To 'screeg' something is to swing a sword at it. An example of a sentence that uses the word screeg is:"
    }   
]

#
print(tokenizer.apply_chat_template(one_shot_prompt, tokenize=False))
#
outputs = pipe(one_shot_prompt)
print(outputs[0]["generated_text"])

# break up the prompt into multiple steps
product_prompt = [
    {"role": "user",
     "content": "Create a name and slogan for my car brand."}
]

outputs = pipe(product_prompt)
#
product_description = outputs[0]["generated_text"]
print(product_description)

# Continue the problem with a new prompt
sales_prompt = [
    {"role": "user",
     "content": f"Generate a sales pitch for: '{product_description}'"}
]
outputs = pipe(sales_prompt)
sales_pitch = outputs[0]["generated_text"]

#
print(sales_pitch)