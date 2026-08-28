from llama_cpp.llama import Llama
import json

# Load Phi-3
llmm = Llama.from_pretrained(
    repo_id = "microsoft/Phi-3-mini-4k-instruct-gguf",
    filename= "*fp16.gguf",
    n_gpu_layers = -1,
    n_ctx = 2048,
    verbose = False
)

# Generate output
output = llmm.create_chat_completion(
    messages=[
        {"role": "user", "content": "Create name, gender and age stats for a random character in JSON format."},
    ],
    response_format={"type": "json_object"},
    temperature=0,
)['choices'][0]['message']["content"]


#
# Format as json
json_output = json.dumps(json.loads(output), indent=4)
print(json_output)