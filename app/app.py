import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd

from src.preprocess import load_dataset
from src.embedder import load_embedding_model, embed_texts
from src.clustering import kmeans_cluster
from src.summarizer import summarize_cluster

# Cache functions
@st.cache_data
def load_data():
    df = load_dataset("data/News_Category_Dataset_v3.json")
    return df

@st.cache_resource
def load_model_cached():
    return load_embedding_model()

@st.cache_data
def compute_embeddings(_model, texts):
    return embed_texts(_model, texts)

@st.cache_data
def perform_clustering(embeddings, k):
    labels, model = kmeans_cluster(embeddings, k)
    return labels

# -------------------- STREAMLIT UI --------------------

st.title("News Clustering & Summarization Prototype")

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

# Initialize session_state for tracking expanded clusters
if 'expanded_clusters' not in st.session_state:
    st.session_state['expanded_clusters'] = {}

# Create buttons for each cluster with minimal, descriptive text
for cluster_id in cluster_ids:
    # The button will only show "Cluster X Summary" (no detailed text from the cluster)
    cluster_summary_text = f"Cluster {cluster_id} Summary [+]"
    
    # Button to show cluster details
    button_key = f"cluster_{cluster_id}_summary"
    
    # Toggle the expand/collapse action on button click
    if st.button(cluster_summary_text, key=button_key, help="Click to show/hide the cluster summary", use_container_width=True):
        # Toggle expanded state
        if cluster_id not in st.session_state['expanded_clusters']:
            st.session_state['expanded_clusters'][cluster_id] = False  # Set default collapsed state

        # Toggle the state when clicked
        st.session_state['expanded_clusters'][cluster_id] = not st.session_state['expanded_clusters'][cluster_id]

    # If the cluster content is expanded, show the summary and headlines
    if st.session_state['expanded_clusters'].get(cluster_id, False):
        # Generate the cluster summary and sample headlines for the selected cluster
        texts = sample_df[sample_df.cluster == cluster_id]['clean_text'].tolist()
        summary = summarize_cluster(texts, num_sentences=3)  # First 3 sentences as a summary

        st.subheader(f"Cluster {cluster_id} Summary")
        st.write(f"**Summary**: {summary}")

        # Get the sample headlines
        headlines = sample_df[sample_df.cluster == cluster_id]['headline'].tolist()

        # Limit the headlines to the top 10
        st.subheader("Top 10 Headlines")
        for h in headlines[:10]:
            st.write(f"- {h}")
