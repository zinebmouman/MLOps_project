import pandas as pd
import re, logging, joblib
import nltk
from pathlib import Path
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "spotify_millsongdata.csv"
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)

def ensure_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

def preprocess_text_factory():
    stop_words = set(stopwords.words('english'))
    pat = re.compile(r"[^a-zA-Z\s]")
    def _clean(text: str) -> str:
        text = pat.sub("", str(text)).lower()
        tokens = word_tokenize(text)
        tokens = [w for w in tokens if w and w not in stop_words]
        return " ".join(tokens)
    return _clean

if __name__ == "__main__":
    logging.info("🚀 Starting preprocessing...")
    ensure_nltk()

    # Charger un échantillon raisonnable pour le build
    df = pd.read_csv(DATA).sample(10000, random_state=42)
    logging.info("✅ Dataset loaded: %d rows", len(df))

    # Retirer colonne 'link' si présente
    df = df.drop(columns=['link'], errors='ignore').reset_index(drop=True)

    # Nettoyage
    logging.info("🧹 Cleaning text...")
    clean_fn = preprocess_text_factory()
    df['cleaned_text'] = df['text'].map(clean_fn)
    logging.info("✅ Text cleaned.")

    # TF-IDF
    logging.info("🔠 Vectorizing TF-IDF...")
    tfidf = TfidfVectorizer(max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df['cleaned_text'])
    logging.info("✅ TF-IDF shape: %s", tfidf_matrix.shape)

    # Similarités cosinus
    logging.info("📐 Computing cosine similarity...")
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Sauvegarde des artefacts dans /models
    joblib.dump(df[['artist','song','cleaned_text']], MODELS / 'df_cleaned.pkl')
    joblib.dump(tfidf,                         MODELS / 'tfidf_vectorizer.pkl')
    joblib.dump(cosine_sim,                   MODELS / 'cosine_sim.pkl')

    logging.info("💾 Saved: df_cleaned.pkl, tfidf_vectorizer.pkl, cosine_sim.pkl")
    logging.info("✅ Preprocessing complete.")
