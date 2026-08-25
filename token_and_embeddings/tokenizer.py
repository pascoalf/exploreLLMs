from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct",
        device_map = "cuda",
        torch_dtype = "auto",
        trust_remote_code = True,
        attn_implementation = "eager"
    )


tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")


# create a pipeline
generator = pipeline(
        "text-generation",
        model = model,
        tokenizer = tokenizer,
        return_full_text = False,
        max_new_tokens = 500,
        do_sample = False
    )

##
newprompt = "What's your name?"

# tokenize the input prompt
input_ids = tokenizer(newprompt, return_tensors = "pt").input_ids.to("cuda")

# generate the text // not the generator
generation_output = model.generate(
    input_ids = input_ids,
    max_new_tokens = 20
)

print(input_ids)

print(generation_output)

print(tokenizer.decode(generation_output[0]))

# to decode each token
for id in input_ids[0]:
    print(tokenizer.decode(id))


print(tokenizer.decode(3323))
print(tokenizer.decode(622))
print(tokenizer.decode([3323, 622]))
print(tokenizer.decode(29901))