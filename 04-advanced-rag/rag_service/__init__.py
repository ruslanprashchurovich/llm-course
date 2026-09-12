"""rag_service — RAG-сервис «Векторики» для модуля 4: надёжность, безопасность,
наблюдаемость.

Карта пакета (какой урок какой модуль разбирает):
    config.py        — настройки из env                      (уроки 4.1+)
    embeddings.py    — TF-IDF по умолчанию / e5 опционально  (урок 4.1)
    store.py         — Qdrant: коллекция, фильтры, upsert    (урок 4.1)
    ingest.py        — ETL индексации с санитизацией         (уроки 4.1, 4.2)
    security.py      — injection-детектор, PII, output guard (урок 4.2)
    access.py        — Principal и фильтр прав               (уроки 4.2, 4.5)
    prompts.py       — наивный и укреплённый промпты         (урок 4.2)
    llm.py           — клиент Ollama + StubLLM               (уроки 4.1+)
    pipeline.py      — полный путь запроса                   (урок 4.2)
    retrieval.py     — dense + BM25 + RRF (+ rerank)         (уроки 4.1, 4.4)
    observability.py — JSON-логи, Prometheus, трейсинг       (урок 4.3)
    app.py           — FastAPI с /ask и /metrics             (урок 4.3)
"""
