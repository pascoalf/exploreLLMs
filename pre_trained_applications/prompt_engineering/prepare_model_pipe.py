"""from https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch06.html#in_context_learning_providing_examples"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model = None
pipe = None
tokenizer = None

def main():
    global model, pipe, tokenizer
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
        max_new_tokens = 100,
        do_sample = False
    )

if __name__ == "__main__":
    main()
