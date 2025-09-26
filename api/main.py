from fastapi import FastAPI, Query
from fastapi.responses import Response, RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from pathlib import Path
import os, json

from src.recommendation import recommend_songs, list_songs

app = FastAPI(title="Music Recommender API", version="1.0.0")

# CORS: autoriser la webapp (Streamlit) à appeler l'API
UI_ORIGIN = os.getenv("UI_ORIGIN", "*")  # par ex: http://localhost:8501
app.add_middleware(
    CORSMiddleware,
    allow_origins=[UI_ORIGIN] if UI_ORIGIN != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REQ_COUNTER = Counter("requests_total", "Total requests", ["endpoint"])
RECO_LAT = Histogram("recommend_latency_seconds", "Latency for /recommend")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", None)
MLFLOW_UI_URL       = os.getenv("MLFLOW_UI_URL", None)

@app.get("/")
def root():
    REQ_COUNTER.labels("/").inc()
    return {"message": "Music Recommender is running. See /docs, /health, /metrics, /mlflow, /version."}

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)

@app.get("/health")
def health():
    REQ_COUNTER.labels("/health").inc()
    return {"status": "ok"}

@app.get("/version")
def version():
    REQ_COUNTER.labels("/version").inc()
    info_path = Path(__file__).resolve().parents[1] / "models" / "model_info.json"
    if info_path.exists():
        return json.loads(info_path.read_text(encoding="utf-8"))
    return {"detail": "model_info.json not found"}

# -------- Reco + listing de chansons --------
@app.get("/songs")
def songs(q: str = "", limit: int = 300):
    """Liste de chansons (recherche contient), limitée pour l'UI."""
    REQ_COUNTER.labels("/songs").inc()
    return {"items": list_songs(q=q, limit=limit)}

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
        return {
            "message": "MLflow UI URL not configured. Set MLFLOW_UI_URL to enable redirect.",
            "tracking_uri": uri_env,
            "experiments": [
                {
                    "name": e.name,
                    "experiment_id": e.experiment_id,
                    "artifact_location": e.artifact_location,
                    "lifecycle_stage": e.lifecycle_stage,
                }
                for e in exps
            ],
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "error": str(e),
            "hint": "Set MLFLOW_TRACKING_URI and/or MLFLOW_UI_URL."
        })
