FROM python:3.11-slim
WORKDIR /app

# Dépendances Python (runtime)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code API + lib
COPY api ./api
COPY src ./src

# Artefacts entraînés (provenant de l'artefact du job CI)
COPY models ./models

EXPOSE 8000
CMD ["uvicorn","api.main:app","--host","0.0.0.0","--port","8000"]


