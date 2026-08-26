from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
from copy import deepcopy
import pandas as pd
from bertopic.representation import KeyBERTInspired
from bertopic.representation import MaximalMarginalRelevance
from transformers import pipeline
from bertopic.representation import TextGeneration

dataset = load_dataset("maartengr/arxiv_nlp")["train"]

# Extract metadata
abstracts = dataset["Abstracts"]
titles = dataset["Titles"]

# Create an embedding for each abstract
embedding_model = SentenceTransformer("thenlper/gte-small")
embeddings = embedding_model.encode(abstracts, show_progress_bar = True)

# UMAP dim. reduction
umap_model = UMAP(
    n_components=5, min_dist=0.0, 
    metric = "cosine", 
    random_state = 42 ## this disables parallel computing
)
reduced_embeddings = umap_model.fit_transform(embeddings)

# HBDSCAN
# fit the model with HDBSCAN
hdbscan_model = HDBSCAN(
    min_cluster_size=50,
    metric = "euclidean",
    cluster_selection_method = "eom"
).fit(reduced_embeddings)

# train model based on previous embedding, umap and clustering
topic_model = BERTopic(
    embedding_model= embedding_model,
    umap_model = umap_model,
    hdbscan_model = hdbscan_model,
    verbose = True
).fit(abstracts, embeddings)

# A quick description can be obtained by get_topic_info()
print(topic_model.get_topic_info())

# to represent the topic in a single word
print(topic_model.get_topic(0))

# To look for a specific topic
print(topic_model.find_topics("topic modeling"))

# copy coriginal representations
original_topics = deepcopy(topic_model.topic_representations_)

# wraper to visualize topic differences 

def topic_differences(model, original_topics, nr_topics):
    """Show the differences in topic representations between two models """
    df = pd.DataFrame(columns = ["Topic", "Original", "Updated"])
    for topic in range(nr_topics):

        # Extract top 5 words per topic per model
        og_words = " | ".join(list(zip(*original_topics[topic]))[0][:5])
        new_words = " | ".join(list(zip(*model.get_topic(topic)))[0][:5])
        df.loc[len(df)] = [topic, og_words, new_words]

    return df


# Update topic representation using KeyBERTInspired
representation_model = KeyBERTInspired()
topic_model.update_topics(abstracts, representation_model = representation_model)

# Show topic differences
print(topic_differences(topic_model, original_topics,  nr_topics = 4))

# Update topic representation to maximal marginal relevance
representation_model = MaximalMarginalRelevance(diversity = 0.2)
topic_model.update_topics(abstracts, representation_model = representation_model)

# Shoow topic differences
print(topic_differences(topic_model, original_topics, nr_topics = 4))


# Add a text generation block
prompt = """I have a topic that contains the following:
[DOCUMENTS]

The topic is described by the following keywords: '[KEYWORDS]'.

Based on the documents and keywords, what is this topic about?"""

# Update our topic representations using Flan-T5
generator = pipeline("text2text-generation",
                     model = "google/flan-t5-small")
representation_model = TextGeneration(
    generator, 
    prompt = prompt, 
    doc_length = 50,
    tokenizer = "whitespace"
)

topic_model.update_topics(abstracts, 
                                representation_model = representation_model)

# show topic differences
print(topic_differences(topic_model, original_topics, nr_topics = 4))

