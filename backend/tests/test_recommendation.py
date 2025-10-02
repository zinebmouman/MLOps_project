import pandas as pd
import numpy as np
from src import recommendation as rec

def _fake_load():
    df = pd.DataFrame(
        {"artist": ["A", "B", "C"], "song": ["Love", "Live", "Lover"]}
    )
    cos = np.array([[1, 0.9, 0.2], [0.9, 1, 0.3], [0.2, 0.3, 1]])
    return df, cos

def test_list_songs_filter_and_limit(monkeypatch):
    monkeypatch.setattr(rec, "_load", _fake_load)
    assert rec.list_songs(q="lo", limit=2) == ["Live", "Love"]  # trié

def test_recommend_found(monkeypatch):
    monkeypatch.setattr(rec, "_load", _fake_load)
    out = rec.recommend_songs("Love", top_n=2)
    assert out is not None
    assert list(out.columns) == ["artist", "song"]
    assert len(out) == 2

def test_recommend_not_found(monkeypatch):
    monkeypatch.setattr(rec, "_load", _fake_load)
    assert rec.recommend_songs("Unknown") is None
