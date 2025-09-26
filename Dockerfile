FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Les modèles sont fournis par le job CI via download-artifact
COPY api ./api
COPY src ./src
# Les modèles sont fournis par le job CI via download-artifact
COPY models ./models

EXPOSE 8000
CMD ["uvicorn","api.main:app","--host","0.0.0.0","--port","8000"]
