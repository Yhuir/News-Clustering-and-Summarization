import pandas as pd
import re

def clean_text(text):
    """
    Args:
        text (_type_): _description_

    Returns:
        _type_: _description_
    """
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def load_dataset(path):
    """ 
    Load dataset from JSONL file and preprocess text.
    Args:
        path (str): Path to the JSONL file.
        Returns:
        pd.DataFrame: DataFrame
        
    """
    df = pd.read_json(path, lines=True)

    # Combine headline + short description
    df['content'] = df['headline'] + ". " + df['short_description']

    # Clean the combined text
    df['clean_text'] = df['content'].apply(clean_text)

    # Keep short_description so we can use it for ROUGE evaluation
    return df[['headline', 'short_description', 'clean_text', 'category']]