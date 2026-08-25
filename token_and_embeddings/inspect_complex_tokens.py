import compare_tokenization as ct
import generate_contextualized_embeddings as context

code_example = """def add_numbers(a, b):
                
                # Add a + b

                return a + b"""

ct.compare_tokens(code_example)

code_context = context.context_embedding(code_example, 
                          tokenizer = "microsoft/deberta-base",
                          model =  "microsoft/deberta-v3-xsmall")

print(code_context)


paragraph = """The preceding guided tour of trained 
            tokenizers showed a number of ways in which
            actual tokenizers differ from each other.
            But what determines their tokenization behavior? 
            There are three major groups of design choices 
            that determine how the tokenizer will break down text: 
            the tokenization method, the initialization parameters, 
            and the domain of the data the tokenizer targets."""

par_context = context.context_embedding(paragraph, 
                          tokenizer = "microsoft/deberta-base",
                          model =  "microsoft/deberta-v3-xsmall")

print(par_context)

print(type(par_context))