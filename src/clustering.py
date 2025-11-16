from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def kmeans_cluster(embeddings, k=8):
    """
    Perform KMeans clustering.
    Returns: labels, model
    """
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(embeddings)
    return labels, km

def evaluate_clusters(embeddings, labels):
    """
    Compute silhouette score (optional).
    """
    if len(set(labels)) < 2:
        return -1  # silhouette can't be computed
    return silhouette_score(embeddings, labels)
