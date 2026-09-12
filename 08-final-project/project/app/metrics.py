"""Метрики Prometheus.

Определяются на уровне модуля: prometheus-client хранит их в глобальном
реестре, и повторная регистрация с тем же именем — ошибка. Экспорт —
через ASGI-приложение, смонтированное на /metrics (см. app/main.py).
"""

from __future__ import annotations

from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Число обработанных HTTP-запросов",
    labelnames=("method", "path", "status"),
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "Длительность обработки HTTP-запросов",
    labelnames=("method", "path"),
)

SEARCH_LATENCY = Histogram(
    "rag_search_duration_seconds",
    "Длительность этапа поиска (эмбеддинг + Qdrant + reranker)",
)

GENERATION_LATENCY = Histogram(
    "rag_generation_duration_seconds",
    "Полная длительность генерации ответа",
    # LLM на CPU медленная: стандартные бакеты (до 10 с) не подходят.
    buckets=(1, 5, 10, 20, 30, 60, 120, 240, float("inf")),
)

GENERATED_TOKENS = Counter(
    "rag_generated_tokens_total",
    "Сгенерированные токены (приблизительно: один чанк стрима Ollama ~ один токен)",
)

LLM_ERRORS = Counter(
    "rag_llm_errors_total",
    "Ошибки обращения к LLM-серверу",
)
