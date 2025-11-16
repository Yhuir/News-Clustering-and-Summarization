# News-Clustering-and-Summarization

This project groups similar news articles and generates concise summaries for each cluster. It uses a combination of K-Means clustering and extractive summarization (TextRank) to help users digest large volumes of news more efficiently.

## How to Run

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download the dataset: Get the "News Category Dataset" from Kaggle:**
    ```bash
    kaggle datasets download rmisra/news-category-dataset
    unzip news-category-dataset.zip -d data/
    ```

3. **Run the Streamlit app:**
    ```bash
    streamlit run app/app.py
    ```

4. **Access the app:** Open your web browser and go to `http://localhost:8501`.

## Project Structure
- `app/`: Contains the Streamlit application code.
- `data/`: Directory for datasets (not included in the repository).
- `src/`: Source code for clustering and summarization.
- `requirements.txt`: List of Python dependencies.
- `README.md`: Project documentation.

## Dependencies
See `requirements.txt` for a full list of dependencies. 

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

