"""
evaluation script for:
1. Clustering metrics 
2. ROUGE scores for extractive summaries vs. short_description
"""

import numpy as np

from src.preprocess import load_dataset
from src.embedder import load_embedding_model, embed_texts
from src.clustering import (
    kmeans_cluster,
    evaluate_all_cluster_metrics,
)
from src.summarizer import summarize_text

from rouge_score import rouge_scorer


def evaluate_clustering(sample_size=1000, k=8, data_path="data/News_Category_Dataset_v3.json"):
    """
    Evaluate clustering quality using multiple metrics on a sample of the dataset.
    Uses KMeans clustering by default.
    """
    print("=== Clustering Evaluation ===")
    df = load_dataset(data_path)
    df_sample = df.sample(sample_size, random_state=42)

    model = load_embedding_model()
    embeddings = embed_texts(model, df_sample["clean_text"].tolist())

    labels, _ = kmeans_cluster(embeddings, k=k)
    metrics = evaluate_all_cluster_metrics(embeddings, labels)

    for name, value in metrics.items():
        print(f"{name}: {value}")


def evaluate_rouge(sample_size=200, data_path="data/News_Category_Dataset_v3.json"):
    """
    Approximate summary quality by comparing extractive summaries of each article
    against the article's short_description using ROUGE.
    """
    print("\n=== ROUGE Evaluation (article-level) ===")
    df = load_dataset(data_path)
    df_sample = df.sample(sample_size, random_state=42)

    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

    scores_rouge1 = []
    scores_rougeL = []

    for _, row in df_sample.iterrows():
        reference = row["short_description"]
        # summary from the cleaned content
        generated = summarize_text(row["clean_text"], num_sentences=3)

        if not reference or not generated:
            continue

        score = scorer.score(reference, generated)
        scores_rouge1.append(score["rouge1"].fmeasure)
        scores_rougeL.append(score["rougeL"].fmeasure)

    if scores_rouge1:
        print(f"Avg ROUGE-1 F1: {np.mean(scores_rouge1):.4f}")
    else:
        print("No valid ROUGE-1 scores computed.")

    if scores_rougeL:
        print(f"Avg ROUGE-L F1: {np.mean(scores_rougeL):.4f}")
    else:
        print("No valid ROUGE-L scores computed.")


if __name__ == "__main__":
    # Run both evaluations
    evaluate_clustering(sample_size=1000, k=8)
    evaluate_rouge(sample_size=200)
