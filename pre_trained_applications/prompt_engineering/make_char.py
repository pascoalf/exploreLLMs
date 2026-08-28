"""from https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch06.html#in_context_learning_providing_examples"""
import prepare_model_pipe as pmp

pmp.main()

# zero-shot -- ask to make a character, but provide no example
zeroshot_promppt = [
    {"role": "user", "content": "Make a character (name and atack stats)"}
    ]

outputs = pmp.pipe(zeroshot_promppt)
print(outputs[0]["generated_text"])

# One shot learning - provide one example
one_shot_template = """Example:

{
  "description": "Wild",
  "name": "Valentino",
  "armor": "Leather",
  "weapon": "Sword"
}
"""
one_shot_prompt = [
    {"role": "user", "content": "Make a character (name and atack stats) like " + one_shot_template}
]

# Generate the output
outputs = pmp.pipe(one_shot_prompt, do_sample=True, top_p=0.4, temperature = 0.6)
print(outputs[0]["generated_text"])