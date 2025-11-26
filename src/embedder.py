from sentence_transformers import SentenceTransformer
import numpy as np

def load_embedding_model():
    """
    A solid default model for clustering + summarization
    """
    return SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

def embed_texts(model, texts, batch_size=64):
    """ 
    Embed a list of texts using the provided SentenceTransformer model.
    Args:
        model (SentenceTransformer): Preloaded embedding model.
        texts (List[str]): List of texts to embed.
        batch_size (int): Batch size for embedding.
        
    Returns:
        np.ndarray: Array of embedding vectors.
    """
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True
    )
    return np.array(embeddings)
