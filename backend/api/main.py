# api/main.py
from fastapi import FastAPI, Query
from fastapi.responses import Response, RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware   # 🔹
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from src.recommendation import recommend_songs
from src.recommendation import _load as _load_model   # 🔹 (pour /songs)
from pathlib import Path
import os, json

app = FastAPI(title="Music Recommender API", version="1.0.0")

# ---- CORS (autorise l'UI) 🔹
UI_ORIGIN = os.getenv("UI_ORIGIN", "http://localhost:8501")
origins = {UI_ORIGIN, "http://127.0.0.1:8501"} if UI_ORIGIN else {"*"}
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(origins),
    allow_methods=["GET"],
    allow_headers=["*"],
)

REQ_COUNTER = Counter("requests_total", "Total requests", ["endpoint"])
RECO_LAT = Histogram("recommend_latency_seconds", "Latency for /recommend")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", None)
MLFLOW_UI_URL       = os.getenv("MLFLOW_UI_URL", None)

@app.get("/")
def root():
    REQ_COUNTER.labels("/").inc()
    return {"message": "Music Recommender is running. See /docs, /health, /metrics, /mlflow."}

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)

@app.get("/health")
def health():
    REQ_COUNTER.labels("/health").inc()
    return {"status": "ok"}

# 🔹 liste de chansons pour l'UI (recherche côté API)
@app.get("/songs")
def songs(q: str = "", limit: int = 500):
    df, _ = _load_model()
    s = df["song"].dropna().astype(str)
    if q:
        s = s[s.str.contains(q, case=False, na=False)]
    items = sorted(s.unique().tolist())[:max(1, min(limit, 5000))]
    return {"count": len(items), "items": items}

@app.get("/version")
def version():
    p = Path(__file__).resolve().parents[1] / "models" / "model_info.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"trained_at": None, "msg": "model_info.json missing"}

@app.get("/recommend")
@RECO_LAT.time()
def recommend(song: str = Query(..., min_length=1), top_n: int = 5):
    REQ_COUNTER.labels("/recommend").inc()
    res = recommend_songs(song, top_n=top_n)
    if res is None:
        return {"song": song, "found": False, "recommendations": []}
    return {
        "song": song,
        "found": True,
        "recommendations": res.to_dict(orient="records")
    }

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# ---------- MLflow access ----------
@app.get("/mlflow")
def mlflow_entry():
    if MLFLOW_UI_URL:
        return RedirectResponse(MLFLOW_UI_URL)

    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        uri_env = MLFLOW_TRACKING_URI or mlflow.get_tracking_uri()
        client = MlflowClient(tracking_uri=uri_env)
        exps = client.search_experiments()
        experiments = [{
            "name": e.name,
            "experiment_id": e.experiment_id,
            "artifact_location": e.artifact_location,
            "lifecycle_stage": e.lifecycle_stage
        } for e in exps]
        return {
            "message": "MLflow UI URL not configured. Set MLFLOW_UI_URL to enable redirect.",
            "tracking_uri": uri_env,
            "experiments": experiments
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "error": str(e),
            "hint": "Set MLFLOW_TRACKING_URI and/or MLFLOW_UI_URL environment variables."
        })
