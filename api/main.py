from fastapi import FastAPI, Query
from fastapi.responses import Response, RedirectResponse, JSONResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from src.recommendation import recommend_songs
from pathlib import Path
import os

app = FastAPI(title="Music Recommender API", version="1.0.0")

REQ_COUNTER = Counter("requests_total", "Total requests", ["endpoint"])
RECO_LAT = Histogram("recommend_latency_seconds", "Latency for /recommend")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", None)
MLFLOW_UI_URL       = os.getenv("MLFLOW_UI_URL", None)  # ex: http://localhost:5000 ou http://mlflow:5000

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
    # 1) Si on connaît l'URL de l'UI, on redirige
    if MLFLOW_UI_URL:
        return RedirectResponse(MLFLOW_UI_URL)

    # 2) Sinon, on expose des infos du tracking (liste des experiments)
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
