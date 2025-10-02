# backend/tests/test_preprocess_with_repo_data.py
import runpy
from pathlib import Path
import pandas as pd
import numpy as np
import types
import pytest

@pytest.mark.slow
def test_preprocess_with_repo_data(monkeypatch):
    real_csv = Path(__file__).resolve().parents[2] / "backend" / "data" / "spotify_millsongdata.csv"
    if not real_csv.exists():
        pytest.skip("dataset not present in repo")

    # On charge UNE FOIS un petit échantillon (50 lignes)
    df_small = pd.read_csv(real_csv, nrows=50)

    # Environnement de preprocess
    monkeypatch.setenv("DATA_PATH", str(real_csv))   # garde le vrai chemin
    monkeypatch.setenv("SAMPLE_SIZE", "50")          # juste pour l’info/MLflow
    monkeypatch.setenv("TFIDF_MAX_FEATURES", "100")

    # ⚠️ court-circuite pd.read_csv **dans src.preprocess** pour ne PAS lire tout le fichier
    monkeypatch.setattr("src.preprocess.pd.read_csv", lambda _p: df_small.copy())

    # Mocks “légers” pour éviter NLTK/TFIDF réels
    fake_sw = types.SimpleNamespace(words=lambda _lang: ["the", "and"])
    monkeypatch.setattr("src.preprocess.stopwords", fake_sw, raising=True)
    monkeypatch.setattr("src.preprocess.word_tokenize", lambda s: s.split())

    class DummyTFIDF:
        def fit_transform(self, texts):
            n = len(texts)
            return np.eye(n)
    monkeypatch.setattr("src.preprocess.TfidfVectorizer", lambda max_features=None: DummyTFIDF())
    monkeypatch.setattr("src.preprocess.cosine_similarity", lambda a, b: np.eye(a.shape[0]))
    monkeypatch.setattr("src.preprocess.joblib.dump", lambda *a, **k: None)

    class DummyMlflow:
        def set_experiment(self, *_): pass
        class start_run:
            def __init__(self, run_name=None): pass
            def __enter__(self): return self
            def __exit__(self, exc_type, exc, tb): pass
        def set_tags(self, *_): pass
        def log_params(self, *_): pass
        def log_metrics(self, *_): pass
        def log_artifacts(self, *_): pass
    monkeypatch.setattr("src.preprocess.mlflow", DummyMlflow())

    # Lance le script comme __main__
    runpy.run_module("src.preprocess", run_name="__main__")
