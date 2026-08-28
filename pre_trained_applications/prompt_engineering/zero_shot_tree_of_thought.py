"""from https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch06.html#in_context_learning_providing_examples"""
import prepare_model_pipe as pmp

pmp.main()

# zero-shot tree-of-thought // made something simpler due to lack of memory
zeroshot_tot_prompt = [
    {"role": "user", 
     "content": """Imagine three art critics evaluating a painting where the sky is green instead of blue. "
     "The question is 'should the sky be painted as blue instead of green?'
     """}
]

# Generate the output
outputs = pipe(zeroshot_tot_prompt)
print(outputs[0]["generated_text"])

