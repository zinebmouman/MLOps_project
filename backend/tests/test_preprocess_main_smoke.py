import runpy
import pandas as pd
import numpy as np
import types

def test_preprocess_main_smoke(tmp_path, monkeypatch):
    csv = tmp_path / "data.csv"
    pd.DataFrame({
        "artist": ["A","B","C"],
        "song": ["Love","Live","Lover"],
        "text": ["hello love", "i live music", "lover hello"]
    }).to_csv(csv, index=False)

    monkeypatch.setenv("DATA_PATH", str(csv))
    monkeypatch.setenv("SAMPLE_SIZE", "3")
    monkeypatch.setenv("TFIDF_MAX_FEATURES", "100")

    # ⚠️ remplace l’objet stopwords au lieu de stopwords.words
    fake_sw = types.SimpleNamespace(words=lambda _lang: ["the", "and"])
    monkeypatch.setattr("src.preprocess.stopwords", fake_sw, raising=True)
    monkeypatch.setattr("src.preprocess.word_tokenize", lambda s: s.split())
    monkeypatch.setattr("src.preprocess.nltk.data.find", lambda *_: None)

    class DummyTFIDF:
        def fit_transform(self, texts):
            return np.eye(len(texts))

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

    runpy.run_module("src.preprocess", run_name="__main__")
