# News-Clustering-and-Summarization

This project groups similar news articles and generates concise summaries for each cluster. It uses sentence embeddings (Sentence-Transformers), clustering (K-Means and DBSCAN), and extractive summarization (TextRank) to help users digest large volumes of news more efficiently.

This README serves as the main documentation for the project.  
It explains both **how to use the software** (installation, running the app, workflow) and **how the system is implemented** (module-by-module description of the pipeline).

## How to Run


### 1. Create & activate a virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate      # macOS / Linux
# venv\Scripts\activate.bat   # Windows (cmd)
# venv\Scripts\Activate.ps1   # Windows (PowerShell)
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```
This will install packages such as sentence-transformers, streamlit, sumy, scikit-learn, and others required for the app.

### 3. Download or prepare your dataset

This project uses the **News Category Dataset** from Kaggle.  
You may download it automatically using the Kaggle CLI (recommended) or manually from the dataset page.

#### Option A — Using Kaggle CLI (recommended)

1. **Install the Kaggle CLI** (if you don’t already have it):
```bash
pip install kaggle
```
2. Set up your Kaggle API credentials:
    - Go to https://www.kaggle.com/account
    - Scroll to API section → click Create New API Token
    - This downloads a file named kaggle.json
    - Place it in:
        - macOS / Linux: ~/.kaggle/kaggle.json
        - Windows: C:\Users\<YourUser>\.kaggle\kaggle.json
    - Ensure correct permissions:

        ```bash
        chmod 600 ~/.kaggle/kaggle.json
        ```

    - Download the dataset into the data/ directory:

        ```bash
        kaggle datasets download rmisra/news-category-dataset
        unzip news-category-dataset.zip -d data/
        ```

    - After unzipping, you should see a file like:
    - data/News_Category_Dataset_v3.json


#### Option B — Manual Download (alternative)
1. Open the dataset page: https://www.kaggle.com/datasets/rmisra/news-category-dataset
2. Click Download
3. Extract the zip file manually
4. Place the JSON file into the data/ directory

### 4. Run the Streamlit app

```bash
streamlit run app/app.py
```
This will start the Streamlit server and open the app in your default web browser.

## App Workflow
1. Select sample size (e.g., 800–2000 articles).
2. Choose clustering algorithm:
    - K-Means: set the number of clusters (k).
    - DBSCAN: set epsilon and minimum samples.
3. The app will:
    - Load and preprocess the dataset.
    - Generate embeddings using a pre-trained SentenceTransformer model.
    - Perform clustering based on user-selected parameters.
    - Display clusters metrics and per-cluster summaries.

Each cluster panel shows:
    - Cluster ID, dominant category, number of articles.
    - Truncated summary with an option to view the full summary.
    - Category distribution bar chart.
    - Top 10 headlines.
    - A button to show a random article (headline, category, and text).

## Evaluation(Optional Offline Script)

Clustering metrics (Silhouette, Davies–Bouldin, Calinski–Harabasz) are displayed directly in the Streamlit interface as the user changes hyperparameters. An additional `evaluation.py` script is provided for optional offline evaluation, where we also compute ROUGE scores for summaries.

You do **not** need to run this to use the app; it is provided for reproducible offline experiments.

There is an evaluation script that can be used to:

- Compute clustering metrics (Silhouette, Davies–Bouldin, Calinski–Harabasz).
- Compute ROUGE scores comparing generated summaries to the `short_description` field.

From the project root:

```bash
python3 evaluation.py
```

This will print metrics such as:
    - Silhouette Score
    - Davies–Bouldin Index
    - Calinski–Harabasz Index
    - Average ROUGE-1 and ROUGE-L F1 scores


## Project Structure
- `app/`: Contains the Streamlit application code.
- `data/`: Directory for datasets (not included in the repository).
- `src/`: Source code for clustering and summarization.
- `requirements.txt`: List of Python dependencies.
- `README.md`: Project documentation.

## How the Software Is Implemented
This section describes how the system is structured internally and how the main components work together.

### Overall Architecture
1. Data loading & preprocessing (src/preprocess.py)
2. Embedding generation (src/embedder.py)
3. Clustering & metrics (src/clustering.py)
4. Summarization (src/summarizer.py)
5. Interactive UI orchestration (app/app.py)
6. Offline evaluation (optional) (evaluation.py)

### Key Components

#### 1. Data Loading & Preprocessing (src/preprocess.py)

1. The dataset is stored in a JSON-lines file (e.g., News_Category_Dataset_v3.json) under data/.

2. load_dataset(path):

    - Reads the JSON-lines file into a Pandas DataFrame.
    - Combines headline and short_description into a single content field:

    - Cleans the combined text using clean_text, which:
        - Strips leading/trailing whitespace.
        - Normalizes internal whitespace.
    - Stores the cleaned text in a clean_text column.
    - Returns a DataFrame with at least:
            - headline
            - short_description
            - clean_text
            - category
    Role in the pipeline: Provides standardized, cleaned text inputs and labels (categories) that are used for embeddings, clustering, and interpretability (dominant category per cluster).    

#### 2. Embedding Generation (src/embedder.py)
1. load_embedding_model():

    - Loads the pre-trained Sentence-Transformers model:
        sentence-transformers/all-mpnet-base-v2

    - This model converts text into dense vector embeddings that capture semantic similarity.

2. embed_texts(model, texts, batch_size=64):

    - Uses the SentenceTransformer model to encode a list of texts into embeddings.
    - Runs in batches (default 64) for efficiency.
    - Returns a NumPy array of shape [num_texts, embedding_dim].

    Role in the pipeline: Converts preprocessed text into a numerical representation so clustering algorithms can operate in vector space.

#### 3. Clustering & Metrics (src/clustering.py)
This module implements both clustering algorithms and several evaluation metrics.

1. K-Means clustering - kmeans_cluster(embeddings, k=8):
    - Uses sklearn.cluster.KMeans with a fixed number of clusters k.
    - Returns:
        - labels: cluster ID for each embedding (0, 1, ..., k-1).
        - km: the fitted KMeans model.

2. DBSCAN clustering - dbscan_cluster(embeddings, eps=0.3, min_samples=5):
    - Uses sklearn.cluster.DBSCAN with:
        - metric="cosine" (more suitable for sentence embeddings).
        - eps: maximum distance (cosine) between points in a neighborhood.
        - min_samples: minimum number of points in a neighborhood to form a dense region.
    - Returns:
        - labels: cluster IDs, where -1 denotes noise / outliers.
        - db: the fitted DBSCAN model.

3. Clustering metrics - evaluate_all_cluster_metrics(embeddings, labels):

Computes global clustering metrics using scikit-learn:
    - Silhouette Score:
        - Measures how well-separated the clusters are.
        - Range: [-1, 1], higher is better.
    - Davies–Bouldin Index (DBI):
        - Ratio of within-cluster scatter to between-cluster separation.
        - Lower is better.
    - Calinski–Harabasz Index (CHI):
        - Ratio of between-cluster dispersion to within-cluster dispersion.
        - Higher is better.

    Also:
- Returns the number of clusters (based on unique labels).
- Handles edge cases where there is only one cluster or only noise.

Role in the pipeline: Groups semantically similar articles and provides quantitative feedback on cluster quality.



#### 4. Summarization (src/summarizer.py)
Summarization is implemented using TextRank (via the sumy library).

1. summarize_text(text, num_sentences=3):

    - Uses PlaintextParser and TextRankSummarizer from sumy.
    - Produces an extractive summary consisting of the top num_sentences sentences from the input text, ranked by importance.
2. summarize_cluster(texts, num_sentences=3):

    - Concatenates all article texts in a cluster into one large string.
    - Calls summarize_text to generate a cluster-level summary.
    - Returns a short extractive summary that captures the main topics of that cluster.

    Role in the pipeline: Produces concise summaries to describe the content of each cluster, helping users quickly understand what a cluster is about without reading every article.

#### 5. Streamlit App & Orchestration (app/app.py)
The Streamlit app ties everything together and exposes an interactive interface.

1. Caching

    - Uses @st.cache_data and @st.cache_resource to:
        - Cache the loaded dataset.
        - Cache the embedding model.
        - Cache computed embeddings and clustering results.
    - This reduces repeated computation when users change sliders.
2. Controls

    - Sample size slider (e.g., 200–2000).
    - Clustering algorithm selectbox:
      "KMeans" or "DBSCAN".
    - Hyperparameter controls:
      For KMeans: number of clusters k.
      For DBSCAN: eps and min_samples.
3. Pipeline execution

    - Sample a subset of articles based on the chosen sample size.
    - Load or reuse the embedding model.
    - Compute embeddings for clean_text.
    - Run the selected clustering algorithm.
    - Compute clustering metrics.
    - Build cluster-level statistics (size, dominant category).
4. Display / visualization

    - Shows global clustering metrics in the main area.
    - For each cluster:
      - Uses a collapsible expander with:
        - Cluster ID, dominant category, and article count.
        - Truncated summary with a “show full summary” option.
        - Category distribution bar chart (top categories in that cluster).
        - Top 10 headlines.
        - A button to show a random article (headline + full text).

Role in the pipeline: Provides an accessible, exploratory UI that reveals both quantitative and qualitative aspects of the clustering and summarization.

#### 6. Offline Evaluation Script (evaluation.py)
This script is optional and is used for reproducible experiments and additional analysis beyond the UI.

1. Clustering evaluation

    - Samples a subset of articles.
    - Loads the embedding model and generates embeddings.
    - Runs K-Means clustering with a chosen k.
    - Calls evaluate_all_cluster_metrics and prints:
      Silhouette Score
      Davies–Bouldin Index
      Calinski–Harabasz Index
2. Summarization evaluation (ROUGE)

    - Uses the dataset’s short_description as a proxy reference summary.
    - Generates summaries for articles using summarize_text.
    - Computes ROUGE-1 and ROUGE-L F1 scores using rouge-score.
    - Prints average ROUGE scores across the sample.

    Role in the pipeline: Provides quantitative evidence for the quality of clustering and summaries, which complements the visual, qualitative exploration in the Streamlit app.


## Limitations and Future Work

### DBSCAN on high-dimensional embeddings

In addition to K-Means, we implemented DBSCAN as a density-based alternative.  
However, when applied to high-dimensional sentence embeddings of news articles, DBSCAN often behaves in a degenerate way:

- For many (`eps`, `min_samples`) settings, **all points are labeled as noise (`-1`)**, meaning DBSCAN does not find any dense region.
- For other settings, DBSCAN finds **one large dense cluster** containing almost all points and very few (or no) smaller clusters.

This happens because:

- The embeddings live in a **high-dimensional space**, where distances between points tend to concentrate and do not form clearly separated dense regions.
- DBSCAN assumes well-defined dense clusters separated by low-density areas.  
  Real-world news articles, especially across many topics, form a more continuous cloud of points with overlapping themes.

For this dataset, K-Means produced more stable and interpretable clusters (e.g., WELLNESS, POLITICS, TRAVEL, SPORTS), while DBSCAN mostly served as an exploratory tool to confirm that the data does not exhibit strong density-based cluster structure. Extending the project with HDBSCAN or alternative clustering methods is left as future work.

### Summarization quality

We use extractive TextRank summarization, which is simple and unsupervised but has some limitations:

- Summaries can be **verbose or generic**, especially when clusters mix several subtopics.
- TextRank sometimes favors well-written but less central sentences.

Future work could explore abstractive summarization or cluster-by-cluster tuning of summary length.

### UI and scalability

The Streamlit app is designed for **interactive exploration on sampled subsets** (e.g., 200–2000 articles).  
For much larger datasets, rendering many clusters and long summaries in the browser can become slow.  
Pagination, lazy loading, or a backend API would be natural next steps to improve scalability.



## Dependencies
See `requirements.txt` for a full list of dependencies. 

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

