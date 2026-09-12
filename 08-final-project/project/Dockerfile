# Docs Assistant. Образ большой (~2 ГБ): внутри PyTorch CPU для эмбеддингов.
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv/app

# Слои с зависимостями кешируются отдельно от кода (правка кода не пересобирает их).
COPY requirements.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

COPY app ./app
COPY scripts ./scripts
COPY data/docs ./data/docs

# Внутри compose-сети соседи доступны по именам сервисов, а не 127.0.0.1.
ENV APP_QDRANT_URL=http://qdrant:6333 \
    APP_OLLAMA_BASE_URL=http://ollama:11434

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
