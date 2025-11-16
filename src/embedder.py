from sentence_transformers import SentenceTransformer
import numpy as np

def load_embedding_model():
    # A solid default model for clustering + summarization
    return SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

def embed_texts(model, texts, batch_size=64):
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True
    )
    return np.array(embeddings)
