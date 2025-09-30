import os
import json
import time
import re
import logging
from hashlib import sha256
from pathlib import Path

import joblib
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import mlflow

ROOT   = Path(__file__).resolve().parents[1]
DATA_ENV = os.getenv("DATA_PATH", str(ROOT / "data" / "spotify_millsongdata.csv"))
DATA   = Path(DATA_ENV) if "://" not in DATA_ENV else Path("/tmp/spotify.csv")
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

TFIDF_MAX_FEATURES = int(os.getenv("TFIDF_MAX_FEATURES", 5000))
SAMPLE_SIZE        = int(os.getenv("SAMPLE_SIZE", 10000))

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')

def ensure_nltk():
    try: nltk.data.find('tokenizers/punkt')
    except LookupError: nltk.download('punkt')
    try: nltk.data.find('tokenizers/punkt_tab')
    except LookupError: nltk.download('punkt_tab')
    try: nltk.data.find('corpora/stopwords')
    except LookupError: nltk.download('stopwords')

def preprocess_text_factory():
    stop_words = set(stopwords.words('english'))
    pat = re.compile(r"[^a-zA-Z\s]")
    def _clean(text: str) -> str:
        text = pat.sub("", str(text)).lower()
        tokens = word_tokenize(text)
        tokens = [w for w in tokens if w and w not in stop_words]
        return " ".join(tokens)
    return _clean

def _sha256(p: Path):
    if not p.exists(): return None
    h = sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""): h.update(chunk)
    return h.hexdigest()

if __name__ == "__main__":
    logging.info("🚀 Starting preprocessing...")
    ensure_nltk()

    # Si DATA_ENV est une URL, télécharge vers /tmp/spotify.csv
    if "://" in DATA_ENV:
        import urllib.request
        logging.info("⬇️ Downloading dataset from %s", DATA_ENV)
        urllib.request.urlretrieve(DATA_ENV, DATA)

    # Charger un échantillon raisonnable pour les builds rapides
    df_full = pd.read_csv(DATA)
    df = df_full.sample(min(SAMPLE_SIZE, len(df_full)), random_state=42)
    logging.info("✅ Dataset loaded: %d rows (sample from %d)", len(df), len(df_full))

    df = df.drop(columns=['link'], errors='ignore').reset_index(drop=True)

    # --- MLflow tracking ---
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "default"))
    with mlflow.start_run(run_name="preprocess-tfidf-cosine"):
        mlflow.set_tags({
            "git_sha": os.getenv("GITHUB_SHA", "local"),
            "branch":  os.getenv("GITHUB_REF_NAME", "local"),
        })
        mlflow.log_params({
            "tfidf_max_features": TFIDF_MAX_FEATURES,
            "sample_size": len(df),
            "dataset_path": DATA_ENV
        })

        # Nettoyage
        logging.info("🧹 Cleaning text...")
        clean_fn = preprocess_text_factory()
        df['cleaned_text'] = df['text'].map(clean_fn)

        # TF-IDF
        logging.info("🔠 Vectorizing TF-IDF...")
        tfidf = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES)
        tfidf_matrix = tfidf.fit_transform(df['cleaned_text'])

        # Similarités cosinus
        logging.info("📐 Computing cosine similarity...")
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # Sauvegardes artefacts
        joblib.dump(df[['artist','song','cleaned_text']], MODELS / 'df_cleaned.pkl')
        joblib.dump(tfidf,                         MODELS / 'tfidf_vectorizer.pkl')
        joblib.dump(cosine_sim,                   MODELS / 'cosine_sim.pkl')

        # Infos modèle (servira à /version)
        info = {
            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_sha": os.getenv("GITHUB_SHA", "local"),
            "dataset_path": DATA_ENV,
            "dataset_sha256": _sha256(DATA),
            "sample_size": int(len(df)),
            "tfidf_max_features": TFIDF_MAX_FEATURES,
            "tfidf_shape": list(tfidf_matrix.shape),
            "cosine_sim_shape": list(cosine_sim.shape),
        }
        (MODELS / "model_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

        # Log MLflow (metrics + artefacts)
        mlflow.log_metrics({
            "vocab_size": len(getattr(tfidf, "vocabulary_", {})),
            "tfidf_n_rows": tfidf_matrix.shape[0],
            "tfidf_n_cols": tfidf_matrix.shape[1],
        })
        mlflow.log_artifacts(str(MODELS), artifact_path="models")

    logging.info("💾 Saved artefacts in %s", MODELS)
    logging.info("✅ Preprocessing complete.")
