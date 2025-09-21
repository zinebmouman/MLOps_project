from pathlib import Path
import joblib
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from src.utils import save_json, MODELS_DIR

if __name__ == "__main__":
    bundle = joblib.load(MODELS_DIR / "model.pkl")
    pipe = bundle["pipeline"]

    X, y = load_iris(return_X_y=True, as_frame=True)
    _, X_val, _, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    y_pred = pipe.predict(X_val)
    metrics = {
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "f1_macro": float(f1_score(y_val, y_pred, average="macro")),
        "report": classification_report(y_val, y_pred, output_dict=True)
    }
    save_json(metrics, MODELS_DIR / "metrics.json")
    print("[evaluate] metrics:", metrics)
