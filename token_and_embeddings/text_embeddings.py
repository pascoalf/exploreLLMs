from sentence_transformers import SentenceTransformer


def make_txt_embedding(text, transformer):
    # Load model
    model = SentenceTransformer(transformer)

    # Convert text to text embeddings
    vector = model.encode(text)

    assert type(vector.shape) is tuple

    return vector

if __name__ == "__main__":
    an_example = make_txt_embedding(text = "This is an example :)", 
                    transformer= "sentence-transformers/all-mpnet-base-v2")
    print(an_example)
    print(type(an_example))
