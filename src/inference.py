from pathlib import Path
import joblib
import numpy as np
from src.utils import MODELS_DIR
_MODEL = None

def load_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = joblib.load(MODELS_DIR / "model.pkl")["pipeline"]
    return _MODEL

def predict_one(features):
    """
    features: list[float] length=4 (sepal_length, sepal_width, petal_length, petal_width)
    """
    model = load_model()
    X = np.array(features, dtype=float).reshape(1, -1)
    y = model.predict(X)[0]
    proba = model.predict_proba(X)[0].tolist()
    return int(y), proba
