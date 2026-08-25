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

# More prompts
promptA = [
        {"role": "user",
         "content": "What are you?"}]
promptB = [
        {"role": "user",
         "content": "Hi"}]
promptC = [
        {"role": "user",
         "content": "Hi, what is your favorite color?"}]


for i in [promptA, promptB, promptC]:
    new_output = generator(i)
    print(new_output[0]["generated_text"])




