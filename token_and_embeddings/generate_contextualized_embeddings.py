"""Generate contextualized word embeddings
        based on https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch02.html#a_language_model_holds_embeddings_for_t
"""
import torch
from transformers import AutoModel, AutoTokenizer

def context_embedding(sentence, tokenizer, model):
    # Load a tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    # Load a language model
    model = AutoModel.from_pretrained(model)

    # Tokenize the sentence
    tokens = tokenizer(sentence, return_tensors='pt')

    # Process the tokens
    output = model(**tokens)[0]

    # check right type of output
    assert isinstance(output, torch.Tensor)

    #
    return output


if __name__ == "__main__":
    #Example
    context_embedding(tokenizer = "microsoft/deberta-base",
                  model =  "microsoft/deberta-v3-xsmall",
                  sentence = "For example, this sentence :)")
    
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