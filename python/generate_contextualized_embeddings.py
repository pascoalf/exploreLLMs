"""Generate contextualized word embeddings
        based on https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch02.html#a_language_model_holds_embeddings_for_t
"""

from transformers import AutoModel, AutoTokenizer

# Load a tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")

# Load a language model
model = AutoModel.from_pretrained("microsoft/deberta-v3-xsmall")

# Tokenize the sentence
tokens = tokenizer('Hello world', return_tensors='pt')

# Process the tokens
output = model(**tokens)[0]

output.shape

# inspect the vectorrs
for token in tokens['input_ids'][0]:
    print(tokenizer.decode(token))

# the representation used by the LLM
print(output)

"""to do:
- function to create context embeddings
- compare different examples"""