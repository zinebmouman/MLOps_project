# === build stage: train the recommender ===
FROM python:3.11-slim AS trainer
WORKDIR /app

# Déps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code + données
COPY src ./src
COPY data ./data

# Entraînement (préprocessing TF-IDF)
RUN python -m src.preprocess

# === runtime stage: serve API ===
FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api ./api
COPY src ./src
COPY --from=trainer /app/models ./models

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
