from src.preprocess import load_dataset
from src.embedder import load_embedding_model, embed_texts
from src.clustering import kmeans_cluster, evaluate_clusters

df = load_dataset("data/News_Category_Dataset_v3.json")

# sample to avoid heavy work
sample = df.head(500)

model = load_embedding_model()
embeddings = embed_texts(model, sample['clean_text'].tolist())

labels, km = kmeans_cluster(embeddings, k=8)

print("Cluster counts:", {i: list(labels).count(i) for i in set(labels)})

score = evaluate_clusters(embeddings, labels)
print("Silhouette Score:", score)
