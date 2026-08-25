"""Example code to use API
This is from https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/part02.html
"""
import openai
import evaluate_performance as ep

# create client 
client = openai.OpenAI(api_key = "XXX") # I don't have a key

def chatgpt_generation(prompt, document, model="gpt-3.5-turbo-0125"):
    """Generate an output based on a prompt and an input document."""
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant."
            },
        {
            "role": "user",
            "content":   prompt.replace("[DOCUMENT]", document)
            }
    ]
    chat_completion = client.chat.completions.create(
      messages=messages,
      model=model,
      temperature=0
    )
    return chat_completion.choices[0].message.content

# Define a prompt template as a base
prompt = """Predict whether the following movie review is a negative or positive review.

    [DOCUMENT]

    If it is positive return 1 and if it is negative return 0. Do not give any other answers.
"""

#Predict the target using GPT
document = "unpretentious, charming, quircky, original"
chatgpt_generation(prompt, document)

# run predictions
predictions = [
    chatgpt_generation(prompt, doc) for doc in tqdm(data["test"]["text"])
]

# Extract predictions
y_pred = [int(pred) for pred in predictions]

# Evaluate performance
ep.evaluate_performance(data["test"]["label"], y_pred)