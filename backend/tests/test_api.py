from fastapi.testclient import TestClient
from api import main as api_main
import pandas as pd
import numpy as np

def _fake_load():
    df = pd.DataFrame(
        {"artist": ["A", "B", "C"], "song": ["Love", "Live", "Lover"], "cleaned_text": ["", "", ""]}
    )
    cos = np.array([[1, 0.9, 0.2], [0.9, 1, 0.3], [0.2, 0.3, 1]])
    return df, cos

def test_health_and_metrics(monkeypatch):
    c = TestClient(api_main.app)
    assert c.get("/health").json()["status"] == "ok"
    r = c.get("/metrics")
    assert r.status_code == 200
    assert "text/plain" in r.headers.get("content-type", "")

def test_songs_endpoint(monkeypatch):
    monkeypatch.setattr(api_main, "_load_model", _fake_load)
    c = TestClient(api_main.app)
    r = c.get("/songs", params={"q": "lo", "limit": 10})
    js = r.json()
    assert js["count"] >= 1
    assert "Love" in js["items"]

def test_recommend_endpoint(monkeypatch):
    # patch le loader utilisé par recommend_songs
    from src import recommendation as rec
    monkeypatch.setattr(rec, "_load", _fake_load)
    c = TestClient(api_main.app)
    r = c.get("/recommend", params={"song": "Love", "top_n": 2})
    js = r.json()
    assert js["found"] is True and len(js["recommendations"]) == 2

def test_mlflow_redirect(monkeypatch):
    # force le chemin "redirect"
    monkeypatch.setattr(api_main, "MLFLOW_UI_URL", "http://example.com/mlflow")
    c = TestClient(api_main.app)
    r = c.get("/mlflow", allow_redirects=False)
    assert r.status_code in (302, 307)
    assert r.headers["location"].startswith("http://example.com/mlflow")
