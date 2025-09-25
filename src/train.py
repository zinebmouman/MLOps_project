# src/train.py
import os, re, logging, joblib
import pandas as pd
import nltk
from pathlib import Path
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.utils import MODELS_DIR, set_seed, save_json

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')

DATA_PATH = Path("data/spotify_millsongdata.csv")
SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE", "10000"))     # pour CI/build rapide
MAX_FEATURES = int(os.getenv("MAX_FEATURES", "5000"))

def preprocess_text(text, _stop):
    text = re.sub(r"[^a-zA-Z\s]", " ", str(text)).lower()
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in _stop and len(w) > 1]
    return " ".join(tokens)

if __name__ == "__main__":
    set_seed(42)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"CSV not found at {DATA_PATH}. Place your dataset under data/.")

    logging.info("Downloading NLTK data (punkt, stopwords) if needed...")
    nltk.download('punkt')
    nltk.download('stopwords')

    logging.info("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    if "link" in df.columns:
        df = df.drop(columns=["link"])

    if SAMPLE_SIZE > 0 and SAMPLE_SIZE < len(df):
        df = df.sample(SAMPLE_SIZE, random_state=42).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    # Champs attendus: 'text', 'song', 'artist'
    if "text" not in df.columns or "song" not in df.columns:
        raise ValueError("CSV must contain at least 'text' and 'song' columns.")

    stop_words = set(stopwords.words('english'))
    logging.info("Cleaning text...")
    df["cleaned_text"] = df["text"].astype(str).apply(lambda t: preprocess_text(t, stop_words))

    logging.info("Vectorizing with TF-IDF...")
    tfidf = TfidfVectorizer(max_features=MAX_FEATURES)
    tfidf_matrix = tfidf.fit_transform(df["cleaned_text"])

    logging.info("Computing cosine similarity...")
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(df, MODELS_DIR / "df_cleaned.pkl")
    joblib.dump(tfidf_matrix, MODELS_DIR / "tfidf_matrix.pkl")
    joblib.dump(cosine_sim, MODELS_DIR / "cosine_sim.pkl")
    joblib.dump(tfidf, MODELS_DIR / "tfidf_vectorizer.pkl")  # utile si tu veux inférer sur nouveau texte

    save_json(
        {
            "rows": int(len(df)),
            "max_features": int(MAX_FEATURES),
            "sample_size": int(SAMPLE_SIZE),
        },
        MODELS_DIR / "metrics.json",
    )

    logging.info("Saved artefacts in %s", MODELS_DIR)
