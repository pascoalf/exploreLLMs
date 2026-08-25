# Load data from Hugging Face
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

dataset = load_dataset("maartengr/arxiv_nlp")["train"]

# Extract metadata
abstracts = dataset["Abstracts"]
titles = dataset["Titles"]

# Create an embedding for each abstract
embedding_model = SentenceTransformer("thenlper/gte-small")
embeddings = embedding_model.encode(abstracts, show_progress_bar = True)

embeddings.shape



"""To do later: make a function to run over multiple models and compare them"""