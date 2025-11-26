import sys
import os

# Ensure src/ is on the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd  # useful for charts

from src.preprocess import load_dataset
from src.embedder import load_embedding_model, embed_texts
from src.clustering import (
    kmeans_cluster,
    dbscan_cluster,
    evaluate_all_cluster_metrics,
)
from src.summarizer import summarize_cluster

# -------------------- CACHED HELPERS --------------------


@st.cache_data
def load_data():
    # Load the dataset
    df = load_dataset("data/News_Category_Dataset_v3.json")
    return df


@st.cache_resource
def load_model_cached():
    # Load the embedding model
    return load_embedding_model()


@st.cache_data
def compute_embeddings(_model, texts):
    # Compute embeddings for a list of texts
    return embed_texts(_model, texts)


@st.cache_data
def perform_clustering(embeddings, k, algorithm="KMeans", eps=0.8, min_samples=5):
    # Perform clustering and return labels
    if algorithm == "KMeans":
        labels, model = kmeans_cluster(embeddings, k)
    else:
        labels, model = dbscan_cluster(embeddings, eps=eps, min_samples=min_samples)
    return labels


# -------------------- STREAMLIT UI --------------------

# Page title and description
st.title("News Clustering & Summarization Prototype")
st.write(
    "This prototype groups similar news articles and generates cluster summaries "
    "using sentence embeddings, clustering, and extractive summarization."
)

# --- Load data ---
df = load_data()

# --- Controls ---
sample_size = st.slider("Select sample size", 200, 2000, 800)

algorithm = st.selectbox("Clustering algorithm", ["KMeans", "DBSCAN"])

# --- Algorithm-specific parameters ---
if algorithm == "KMeans":
    k = st.slider("Number of clusters (k)", 3, 12, 6)
    eps = None
    min_samples = None
else:
    st.info("DBSCAN does not use k; it discovers the number of clusters automatically.")
    eps = st.slider("DBSCAN eps", 0.05, 1.5, 0.4, step=0.05)
    min_samples = st.slider("DBSCAN min_samples", 3, 20, 5)

    k = None  # not used

# --- Sampling & Embeddings ---
sample_df = df.sample(sample_size, random_state=42).copy()

model = load_model_cached()

# -- Compute embeddings ---
st.write("Generating embeddings... (takes a few seconds)")
embeddings = compute_embeddings(model, sample_df["clean_text"].tolist())

# --- Clustering ---
st.write("Clustering articles...")
labels = perform_clustering(
    embeddings,
    k if k is not None else 0,
    algorithm=algorithm,
    eps=eps if eps is not None else 0.8,
    min_samples=min_samples if min_samples is not None else 5,
)

sample_df["cluster"] = labels

# --- Clustering metrics ---
metrics = evaluate_all_cluster_metrics(embeddings, labels)

st.subheader("Clustering Metrics")
st.write(f"Number of clusters: {metrics['num_clusters']}")

unique_labels = set(labels)

# --- Special messages for DBSCAN ---
if algorithm == "DBSCAN":
    if unique_labels == {-1}:
        # All points are noise
        st.warning(
            "DBSCAN marked all points as noise (cluster -1). "
            "Try increasing eps or decreasing min_samples, "
            "or switch to KMeans."
        )
    elif len(unique_labels) == 1:
        # Exactly one non-noise cluster (e.g., {0})
        st.info(
            "DBSCAN found a single dense cluster. "
            "Try reducing eps or increasing min_samples to get more structure."
        )


# --- Display metrics ---
if metrics["silhouette_score"] is not None:
    st.write(f"Silhouette Score: {metrics['silhouette_score']:.3f}")
else:
    st.write("Silhouette Score: not defined (need at least 2 non-noise clusters).")

if metrics["davies_bouldin_index"] is not None:
    st.write(f"Davies-Bouldin Index: {metrics['davies_bouldin_index']:.3f}")

if metrics["calinski_harabasz_index"] is not None:
    st.write(f"Calinski-Harabasz Index: {metrics['calinski_harabasz_index']:.1f}")

# --- Cluster stats for nicer labels ---
cluster_ids = sorted(sample_df["cluster"].unique())

# Precompute cluster stats
cluster_stats = []
for cid in cluster_ids:
    sub = sample_df[sample_df["cluster"] == cid]
    size = len(sub)
    # simple label: most frequent category in this cluster
    if size > 0:
        top_category = sub["category"].value_counts().idxmax()
    else:
        top_category = "N/A"
    cluster_stats.append(
        {
            "cluster_id": cid,
            "size": size,
            "top_category": top_category,
        }
    )

# --- Cluster summaries ---
st.markdown("---")
st.subheader("Cluster Summaries")

# --- Per-cluster UI using expanders ---
for cluster_id in cluster_ids:
    stats = next(cs for cs in cluster_stats if cs["cluster_id"] == cluster_id)
    sub = sample_df[sample_df["cluster"] == cluster_id]

    cluster_name = f"Cluster {cluster_id}"
    if algorithm == "DBSCAN" and cluster_id == -1:
        cluster_name = "Noise / Outliers"

    header_text = (
        f"{cluster_name} – {stats['top_category']} "
        f"({stats['size']} articles)"
    )

    # Expandable section per cluster    
    with st.expander(header_text, expanded=False):
        texts = sub["clean_text"].tolist()
        summary = summarize_cluster(texts, num_sentences=3)

        # Truncate long summaries for readability
        max_len = 400  # characters
        if isinstance(summary, str) and len(summary) > max_len:
            short_summary = summary[:max_len] + "..."
        else:
            short_summary = summary

        st.markdown(f"**Summary (truncated):** {short_summary}")

        if isinstance(summary, str) and len(summary) > max_len:
            with st.expander("Show full summary"):
                st.write(summary)

        # Category distribution within this cluster
        st.markdown("**Category distribution in this cluster:**")
        if len(sub) > 0:
            cat_counts = sub["category"].value_counts().reset_index()
            cat_counts.columns = ["category", "count"]
            st.bar_chart(cat_counts.set_index("category"))
        else:
            st.write("No articles in this cluster.")

        # Top headlines
        st.subheader("Top 10 Headlines")
        headlines = sub["headline"].tolist()
        for h in headlines[:10]:
            st.write(f"- {h}")

        # show a random article for qualitative inspection
        if st.button(
            "Show a random article from this cluster",
            key=f"random_{cluster_id}",
        ):
            rand_row = sub.sample(1).iloc[0]
            st.markdown(f"**Random article headline:** {rand_row['headline']}")
            st.markdown(f"**Category:** {rand_row['category']}")
            st.markdown(f"**Text:** {rand_row['clean_text']}")
