# === build (trainer): génère models/ ===
FROM python:3.11-slim AS trainer
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY data ./data
ENV TFIDF_MAX_FEATURES=5000 SAMPLE_SIZE=10000
RUN python -m src.preprocess

# === runtime: API ===
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY api ./api
COPY src ./src
COPY --from=trainer /app/models ./models
EXPOSE 8000
CMD ["uvicorn","api.main:app","--host","0.0.0.0","--port","8000"]
