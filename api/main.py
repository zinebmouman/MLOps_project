from fastapi import FastAPI
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from src.schema import PredictRequest, PredictResponse
from src.inference import predict_one, load_model

app = FastAPI(title="Iris ML API", version="1.0.0")

REQ_COUNTER = Counter("requests_total", "Total requests", ["endpoint"])
PRED_LAT = Histogram("predict_latency_seconds", "Latency for /predict")

@app.get("/health")
def health():
    load_model()
    REQ_COUNTER.labels("/health").inc()
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
@PRED_LAT.time()
def predict(body: PredictRequest):
    REQ_COUNTER.labels("/predict").inc()
    label, proba = predict_one(body.as_list())
    return PredictResponse(label=label, proba=proba)

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
