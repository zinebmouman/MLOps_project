# === build (trainer) : génère models/ ===
FROM python:3.11-slim AS trainer
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code + données nécessaires à l'entraînement
COPY src ./src
COPY data ./data

# (facultatif) bornes pour accélérer le build
ENV TFIDF_MAX_FEATURES=5000 SAMPLE_SIZE=10000
# Entraînement / préprocessing TF-IDF
RUN python -m src.preprocess

# === runtime : API ===
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api ./api
COPY src ./src
# On récupère les artefacts du stage trainer
COPY --from=trainer /app/models ./models

EXPOSE 8000
CMD ["uvicorn","api.main:app","--host","0.0.0.0","--port","8000"]
