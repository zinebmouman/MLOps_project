from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib
from pathlib import Path
from src.utils import set_seed, MODELS_DIR

if __name__ == "__main__":
    set_seed(42)
    X, y = load_iris(return_X_y=True, as_frame=True)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=200, n_jobs=1))
    ])
    pipe.fit(X_train, y_train)

    out = MODELS_DIR / "model.pkl"
    joblib.dump({"pipeline": pipe}, out)
    print(f"[train] saved: {out}")
