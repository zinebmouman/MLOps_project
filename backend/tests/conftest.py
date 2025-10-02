import pytest
import pandas as pd
import numpy as np
from src import recommendation as rec

# Fixture auto-appliquée : remplace le chargement du "gros" modèle
@pytest.fixture(autouse=True)
def fake_model(monkeypatch):
    df = pd.DataFrame({
        "artist": ["A", "B", "C"],
        "song":   ["Love", "Life", "Light"],
        "cleaned_text": ["love", "life", "light"],  # au cas où
    })
    cos = np.eye(3)  # matrice identité pour des similarités simples
    monkeypatch.setattr(rec, "_load", lambda: (df, cos))
