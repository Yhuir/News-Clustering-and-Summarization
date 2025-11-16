from src.preprocess import load_dataset
from src.embedder import load_embedding_model, embed_texts

df = load_dataset("data/News_Category_Dataset_v3.json")

model = load_embedding_model()

# Test only on first 100 samples to avoid long wait
sample = df['clean_text'].head(100).tolist()

emb = embed_texts(model, sample)

print("Embedding shape:", emb.shape)
