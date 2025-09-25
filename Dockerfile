# Dockerfile
FROM python:3.11-slim
WORKDIR /app

# Dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code API + librairie
COPY api ./api
COPY src ./src

# ⬇️ Artefacts entraînés par la CI (download-artifact) — DOIT exister au build CI
COPY models ./models

EXPOSE 8000
CMD ["uvicorn","api.main:app","--host","0.0.0.0","--port","8000"]

