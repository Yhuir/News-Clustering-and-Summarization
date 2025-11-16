import sys
import os

# Add project root directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import streamlit as st
import pandas as pd

from src.preprocess import load_dataset
from src.embedder import load_embedding_model, embed_texts
from src.clustering import kmeans_cluster
from src.summarizer import summarize_cluster

@st.cache_data
def load_data():
    df = load_dataset("data/News_Category_Dataset_v3.json")
    return df

@st.cache_resource
def load_model_cached():
    return load_embedding_model()

@st.cache_data
def compute_embeddings(_model, texts):
    embeddings = _model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True
    )
    return embeddings


@st.cache_data
def perform_clustering(embeddings, k):
    labels, model = kmeans_cluster(embeddings, k)
    return labels

# -------------------- STREAMLIT UI --------------------

st.title("📰 News Clustering & Summarization Prototype")

st.write("This prototype groups similar news articles and generates cluster summaries.")

df = load_data()

sample_size = st.slider("Select sample size", 200, 2000, 800)
k = st.slider("Number of clusters", 3, 12, 6)

sample_df = df.sample(sample_size, random_state=42).copy()

model = load_model_cached()

st.write("Generating embeddings... (takes a few seconds)")
embeddings = compute_embeddings(model, sample_df['clean_text'].tolist())

st.write("Clustering articles...")
labels = perform_clustering(embeddings, k)

sample_df['cluster'] = labels

cluster_ids = sorted(sample_df['cluster'].unique())

selected_cluster = st.selectbox("Choose a cluster to explore", cluster_ids)

texts = sample_df[sample_df.cluster == selected_cluster]['clean_text'].tolist()
headlines = sample_df[sample_df.cluster == selected_cluster]['headline'].tolist()

st.subheader("📌 Cluster Summary")
summary = summarize_cluster(texts, num_sentences=3)
st.write(summary)

st.subheader("📰 Sample Headlines")
for h in headlines[:10]:
    st.write("- " + h)
