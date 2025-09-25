from fastapi import FastAPI, Query
from fastapi.responses import Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from src.recommendation import recommend_songs

app = FastAPI(title="Music Recommender API", version="1.0.0")

REQ_COUNTER = Counter("requests_total", "Total requests", ["endpoint"])
RECO_LAT = Histogram("recommend_latency_seconds", "Latency for /recommend")

@app.get("/")
def root():
    REQ_COUNTER.labels("/").inc()
    return {"message": "Music Recommender is running. See /docs, /health, /metrics."}

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
