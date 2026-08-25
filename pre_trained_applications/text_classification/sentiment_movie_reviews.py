from datasets import load_dataset
from transformers import pipeline
import numpy as np
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset
from sklearn.metrics import classification_report


# Load data from Rotten Tomatoes
data = load_dataset("rotten_tomatoes")

print(data)

print(data["train"][120, -1])

# path to HF model selected
model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# load model into pipeline
pipe = pipeline(
    model = model_path,
    tokenizer = model_path,
    return_all_scores = True,
    device = "cuda:0"
)

# Run inference
y_pred = []
for output in tqdm(pipe(KeyDataset(data["test"], "text")), total = len(data["test"])):
    negative_score = output[0]["score"]
    positive_score = output[2]["score"]
    assignment = np.argmax([negative_score,
                            positive_score])
    y_pred.append(assignment)


# To evalute performance
def evaluate_performance(y_true, y_pred):
    """Create and print the classification repoort"""
    performance = classification_report(
        y_true, y_pred,
        target_names=["Negative Review", "Positive Review"]
    )
    print(performance)

# obtai performance report
evaluate_performance(data["test"]["label"], y_pred)

