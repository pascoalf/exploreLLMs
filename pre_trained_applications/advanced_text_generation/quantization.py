from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate

# Load local GGUF model
llm = LlamaCpp(
    model_path="../../Phi-3-mini-4k-instruct-fp16.gguf",
    n_gpu_layers=-1,
    max_tokens=500,
    n_ctx=2048,
    seed=42,
    verbose=False
)

# Generate output directly
print(llm.invoke("Hi! What's your name?"))

# Prompt template
template = """<|user|>
{input_prompt}<|end|>
<|assistant|>"""

prompt = PromptTemplate(
    template=template,
    input_variables=["input_prompt"]
)

# Chain prompt -> model
basic_chain = prompt | llm

response = basic_chain.invoke(
    {
        "input_prompt": "Hi! What is your name?",
    }
)

print(response)