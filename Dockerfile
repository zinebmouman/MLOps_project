# === build stage: train the model ===
FROM python:3.11-slim AS trainer
WORKDIR /app

# 1) Déps Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2) Code
COPY src ./src
ENV PYTHONPATH=/app
# 3) Entraînement + évaluation (sans make)
#   => appelle directement les modules Python
RUN python -m src.train && python -m src.evaluate

# === runtime stage: serve API avec le modèle ===
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api ./api
COPY src ./src
COPY --from=trainer /app/models ./models

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
