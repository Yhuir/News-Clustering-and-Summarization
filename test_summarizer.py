from src.preprocess import load_dataset
from src.embedder import load_embedding_model, embed_texts
from src.clustering import kmeans_cluster
from src.summarizer import summarize_cluster

df = load_dataset("data/News_Category_Dataset_v3.json")
sample = df.head(300)

model = load_embedding_model()
embeddings = embed_texts(model, sample['clean_text'].tolist())

labels, km = kmeans_cluster(embeddings, k=5)
sample['cluster'] = labels

# Test summarization for one cluster
cluster_id = 0
texts = sample[sample.cluster == cluster_id]['clean_text'].tolist()

summary = summarize_cluster(texts, num_sentences=3)

print("Summary for cluster", cluster_id)
print(summary)
