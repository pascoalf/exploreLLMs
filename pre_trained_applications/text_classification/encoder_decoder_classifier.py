from transformers import pipeline
from datasets import load_dataset
import evaluate_performance as ep
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset

# load model (pretrained Flan-T5)
pipe = pipeline(
    "text2text-generation",
    model = "google/flan-t5-small",
    device = "cuda:0"
)

# Load data from Rotten Tomatoes
data = load_dataset("rotten_tomatoes")

# Prepare data and prompt
prompt = "Is the sentence positive or negative?"
data = data.map(lambda example: {"t5": prompt + example['text']})
print(data)

# Run inference
y_pred = []

for output in tqdm(pipe(KeyDataset(data["test"], "t5")), total=len(data["test"])):
    text = output[0]["generated_text"]
    y_pred.append(0 if text == "negative" else 1)


ep.evaluate_performance(data["test"]["label"], y_pred)