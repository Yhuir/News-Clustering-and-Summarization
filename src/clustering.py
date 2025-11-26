from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)


def kmeans_cluster(embeddings, k=8):
    """
    Perform KMeans clustering.
    Returns: labels, model
    """
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(embeddings)
    return labels, km


def dbscan_cluster(embeddings, eps=0.3, min_samples=5):
    """
    Perform DBSCAN clustering on embedding vectors.

    Uses cosine distance (1 - cosine_similarity), which is usually
    more appropriate for sentence embeddings than Euclidean distance.
    """
    db = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric="cosine",   # <- THIS is important
        n_jobs=-1,
    )
    labels = db.fit_predict(embeddings)
    return labels, db


def evaluate_clusters_silhouette(embeddings, labels):
    """
    Compute silhouette score (for backward compatibility).
    """
    if len(set(labels)) < 2:
        return None  # silhouette can't be computed with 1 cluster
    return silhouette_score(embeddings, labels)


def evaluate_all_cluster_metrics(embeddings, labels):
    """
    Compute multiple clustering metrics:
    - Silhouette Score
    - Davies-Bouldin Index
    - Calinski-Harabasz Index

    Returns a dict of metric_name -> value (or None if undefined).
    """
    unique_labels = set(labels)
    metrics = {
        "silhouette_score": None,
        "davies_bouldin_index": None,
        "calinski_harabasz_index": None,
        "num_clusters": len(unique_labels),
    }

    # DBSCAN can produce a single cluster or all noise; guard these cases.
    if len(unique_labels) < 2:
        return metrics

    metrics["silhouette_score"] = silhouette_score(embeddings, labels)
    metrics["davies_bouldin_index"] = davies_bouldin_score(embeddings, labels)
    metrics["calinski_harabasz_index"] = calinski_harabasz_score(embeddings, labels)

    return metrics
