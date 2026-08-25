from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
import evaluate_performance as ep
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity #for zero shot classification

# Load model
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# Load data from Rotten Tomatoes
data = load_dataset("rotten_tomatoes")

# Convert text to embeddings
train_embeddings = model.encode(data["train"]["text"], show_progress_bar = True)
test_embeddings = model.encode(data["test"]["text"], show_progress_bar = True)

print(train_embeddings.shape)

# Train a logistic regression on our train embeddings
clf = LogisticRegression(random_state = 42)
clf.fit(train_embeddings, data["train"]["label"])

# Calculate predictions
y_pred = clf.predict(test_embeddings)
ep.evaluate_performance(data["test"]["label"], y_pred)

"""Zero shot classification
We can tell what the label should mean
Instead of bolean
"""
label_embeddings = model.encode(["A negative review", "A positive review"])

# Find the best matching label for each document
sim_matrix = cosine_similarity(test_embeddings, label_embeddings)
y_pred = np.argmax(sim_matrix, axis = 1)

print("Zero shot below")
print(ep.evaluate_performance(data["test"]["label"], y_pred))