# Load data from Hugging Face
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = load_dataset("maartengr/arxiv_nlp")["train"]

# Extract metadata
abstracts = dataset["Abstracts"]
titles = dataset["Titles"]

# Create an embedding for each abstract
embedding_model = SentenceTransformer("thenlper/gte-small")
embeddings = embedding_model.encode(abstracts, show_progress_bar = True)

print(embeddings.shape)

# reduce the input embeddings from 384 dimensions to 5 dimensions
umap_model = UMAP(
    n_components=5, min_dist=0.0, 
    metric = "cosine", 
    random_state = 42 ## this disables parallel computing
)
reduced_embeddings = umap_model.fit_transform(embeddings)

print(type(reduced_embeddings))
print(type(umap_model))
print(type(UMAP))

# fit the model with HDBSCAN
hdbscan_model = HDBSCAN(
    min_cluster_size=50,
    metric = "euclidean",
    cluster_selection_method = "eom"
).fit(reduced_embeddings)

# Obtain the clusters
clusters = hdbscan_model.labels_

# Number of clusters obtained
len(set(clusters))

# Explore documents in cluster
cluster = 0
for index in np.where(clusters == cluster)[0][:3]:
    print(abstracts[index][:300] + "... \n")

"""For visualization we can redduce dimensions further to 2 dimensions"""
reduced_embeddings2 = UMAP(
    n_components = 2,
    min_dist = 0.0,
    metric = "cosine",
    random_state = 42
).fit_transform(embeddings)

# into a dataframe
df = pd.DataFrame(reduced_embeddings2, columns = ["x", "y"])
df["title"] = titles
df["cluster"] = [str(c) for c in clusters]

# select outliers and non-outliers (clusters)
to_plot = df.loc[df.cluster != "-1", :]
outliers = df.loc[df.cluster == "-1", :]

# Plot outliers and non-outliers separately
plt.scatter(outliers.x,
            outliers.y,
            alpha = 0.05,
            s = 2,
            c = "grey")
plt.scatter(to_plot.x, to_plot.y,
            c = to_plot.cluster.astype(int),
            alpha = 0.6, s = 2, cmap = "tab20b")

"""To do later: make a function to run over multiple models and compare them"""