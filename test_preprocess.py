from src.preprocess import load_dataset

df = load_dataset("data/News_Category_Dataset_v3.json")
print(df.head())
print(len(df))
