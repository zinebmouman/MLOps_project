import os
import json
import time
import re
import logging
from hashlib import sha256
from pathlib import Path
from datetime import datetime
import hashlib
import json

import joblib
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import mlflow
import mlflow.sklearn

# --- Config projet ---
ROOT   = Path(__file__).resolve().parents[1]
DATA_ENV = os.getenv("DATA_PATH", str(ROOT / "data" / "spotify_millsongdata.csv"))
DATA   = Path(DATA_ENV) if "://" not in DATA_ENV else Path("/tmp/spotify.csv")
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

TFIDF_MAX_FEATURES = int(os.getenv("TFIDF_MAX_FEATURES", 5000))
SAMPLE_SIZE        = int(os.getenv("SAMPLE_SIZE", 10000))

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')


# --- Helpers ---
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

def load_new_songs(new_folder: Path) -> pd.DataFrame:
    new_files = list(new_folder.glob("*.csv"))
    if not new_files:
        return pd.DataFrame(columns=['artist','song','text'])
    df_list = [pd.read_csv(f) for f in new_files]
    return pd.concat(df_list, ignore_index=True)

def file_sha256(path: Path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()

# --- Main ---
if __name__ == "__main__":
    start_time = time.time()
    logging.info("🚀 Starting preprocessing...")
    ensure_nltk()

    # Dossier des nouveaux fichiers
    NEW_FOLDER = ROOT / "data" / "new"
    df_new = load_new_songs(NEW_FOLDER)

    # Charger dataset existant
    if "://" in DATA_ENV:
        import urllib.request
        logging.info("⬇️ Downloading dataset from %s", DATA_ENV)
        urllib.request.urlretrieve(DATA_ENV, DATA)

    df_full = pd.read_csv(DATA)

    # Ajouter les nouvelles chansons
    if not df_new.empty:
        logging.info(f"📥 Found {len(df_new)} new songs. Adding to dataset.")
        df_full = pd.concat([df_full, df_new], ignore_index=True)

    # Vérifier hash
    dataset_hash_path = MODELS / "dataset_hash.txt"
    current_hash = _sha256(DATA)
    previous_hash = dataset_hash_path.read_text() if dataset_hash_path.exists() else None

    if previous_hash == current_hash and df_new.empty:
        logging.info("✅ Dataset unchanged. Skipping preprocessing.")
        exit(0)

    logging.info("♻️ Dataset updated. Running preprocessing...")

    # Sample rapide
    df = df_full.sample(min(SAMPLE_SIZE, len(df_full)), random_state=42)
    logging.info("✅ Dataset loaded: %d rows (sample from %d)", len(df), len(df_full))
    df = df.drop(columns=['link'], errors='ignore').reset_index(drop=True)

    # --- Gestion des nouveaux fichiers avec backup ---
BACKUP_FOLDER = ROOT / "data" / "backup"
BACKUP_FOLDER.mkdir(parents=True, exist_ok=True)

backup_hashes = {}

for f in NEW_FOLDER.glob("*.csv"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = BACKUP_FOLDER / f"{timestamp}_{f.name}"
    f.replace(backup_file)  # déplace vers le backup
    logging.info("🗄 Backed up file: %s", backup_file.name)
    backup_hashes[backup_file.name] = file_sha256(backup_file)

# Sauvegarde des hash
with open(BACKUP_FOLDER / "backup_hashes.json", "w") as out:
    json.dump(backup_hashes, out, indent=2)
logging.info("🔑 Saved backup file hashes")

# --- MLflow setup ---
MLFLOW_DIR = ROOT / "mlruns"  # dossier unique pour tracking + artefacts
mlflow.set_tracking_uri(str(MLFLOW_DIR.as_uri()))

exp_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "default")
mlflow.set_experiment(exp_name)  # MLflow crée automatiquement l’experiment si nécessaire

# --- MLflow Run ---
"""
MLflow crée deux types de dossiers :

  Tracking store → métadonnées des runs

  Artifacts → fichiers générés/loggés
"""
with mlflow.start_run(run_name="preprocess-tfidf-cosine") as run:
    logging.info(f"📌 MLflow tracking URI: {mlflow.get_tracking_uri()}")
    logging.info(f"📌 MLflow run ID: {run.info.run_id}")

    # Tags et paramètres
    mlflow.set_tags({
        "git_sha": os.getenv("GITHUB_SHA", "local"),
        "branch": os.getenv("GITHUB_REF_NAME", "local"),
    })
    mlflow.log_params({
        "tfidf_max_features": TFIDF_MAX_FEATURES,
        "sample_size": len(df),
        "dataset_path": DATA_ENV
    })

    # Préprocessing
    logging.info("🧹 Cleaning text...")
    clean_fn = preprocess_text_factory()
    df['cleaned_text'] = df['text'].map(clean_fn)

    logging.info("🔠 Vectorizing TF-IDF...")
    tfidf = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES)
    tfidf_matrix = tfidf.fit_transform(df['cleaned_text'])

    logging.info("📐 Computing cosine similarity...")
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Sauvegarde locale
    joblib.dump(df[['artist','song','cleaned_text']], MODELS / 'df_cleaned.pkl')
    joblib.dump(tfidf,                         MODELS / 'tfidf_vectorizer.pkl')
    joblib.dump(cosine_sim,                   MODELS / 'cosine_sim.pkl')

    # Infos modèle
    info = {
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_sha256": _sha256(DATA),
        "sample_size": len(df),
        "new_songs": len(df_new),
        "total_songs": len(df),
        "unique_artists": df['artist'].nunique(),
        "unique_songs": df['song'].nunique(),
        "tfidf_max_features": TFIDF_MAX_FEATURES,
        "tfidf_shape": list(tfidf_matrix.shape),
        "cosine_sim_shape": list(cosine_sim.shape),
    }
    (MODELS / "model_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    # MLflow metrics
    mlflow.log_metrics({
        "vocab_size": len(getattr(tfidf, "vocabulary_", {})),
        "tfidf_n_rows": tfidf_matrix.shape[0],
        "tfidf_n_cols": tfidf_matrix.shape[1],
        "cosine_sim_rows": cosine_sim.shape[0],
        "cosine_sim_cols": cosine_sim.shape[1],
        "unique_artists": df['artist'].nunique(),
        "unique_songs": df['song'].nunique(),
        "preprocessing_time_sec": round(time.time() - start_time, 2),
        "new_songs": len(df_new),
        "total_songs": len(df)
    })

    # MLflow artefacts (log uniquement les fichiers nécessaires)
    mlflow.log_artifact(MODELS / "df_cleaned.pkl", artifact_path="models")
    mlflow.log_artifact(MODELS / "tfidf_vectorizer.pkl", artifact_path="models")
    mlflow.log_artifact(MODELS / "cosine_sim.pkl", artifact_path="models")
    mlflow.log_artifact(MODELS / "model_info.json", artifact_path="models")

logging.info("💾 Saved artefacts in %s", MODELS)
logging.info("✅ Preprocessing complete.")
