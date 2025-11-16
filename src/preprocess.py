import pandas as pd
import re

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def load_dataset(path):
    # Load the JSON lines file
    df = pd.read_json(path, lines=True)

    # Combine headline + short description
    df['content'] = df['headline'] + ". " + df['short_description']

    # Clean the combined text
    df['clean_text'] = df['content'].apply(clean_text)

    # Keep only what we need
    return df[['headline', 'clean_text', 'category']]
