from pathlib import Path
import joblib

def test_model_exists():
    assert Path("models/model.pkl").exists()

def test_model_loads_and_predicts():
    bundle = joblib.load("models/model.pkl")
    pipe = bundle["pipeline"]
    y = pipe.predict([[5.1, 3.5, 1.4, 0.2]])
    assert y.shape == (1,)
