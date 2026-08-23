import token_methods as tk

# compare tokenization methods

tokenizer_names = ["bert-base-uncased",
                "gpt2",
                "google/flan-t5-small",
                "bigcode/starcoder2-3b",
                "facebook/galactica-125m",
                "microsoft/Phi-3-mini-4k-instruct"]

# To compare different tokenization methods
def compare_tokens(sentence):
    for i in tokenizer_names:
        tk.show_tokens(sentence,
                        tokenizer_name=i)
        print("Method " + i)



if __name__ == "__main__":
    for i in tokenizer_names:
        tk.show_tokens("Olá, sou o Francisco!",
                        tokenizer_name=i)
        print("Method " + i)




