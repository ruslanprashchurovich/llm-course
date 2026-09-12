"""Наблюдаемость RAG-сервиса: структурные логи, метрики Prometheus, трейсинг стадий.

Три «столпа» здесь разделены сознательно:
  * логи   — что произошло в конкретном запросе (высокая детализация, дорого хранить);
  * метрики — агрегаты по всем запросам (дёшево, идут в алерты и дашборды);
  * трейсы — сколько времени заняла каждая стадия одного запроса.

Всё связывается одним trace_id, который живёт в contextvars (как request-scoped
переменная), поэтому его не нужно протаскивать параметром через десять функций.
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import sys
import time
import uuid
from contextlib import contextmanager
from typing import Any, Iterator

# --------------------------------------------------------------------------
# Метрики: работаем и без prometheus_client (тогда метрики — заглушки)
# --------------------------------------------------------------------------
try:  # pragma: no cover - зависит от окружения
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        REGISTRY,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:  # pragma: no cover
    PROMETHEUS_AVAILABLE = False
    CONTENT_TYPE_LATEST = "text/plain"
    REGISTRY = None  # type: ignore[assignment]

    class _NullMetric:
        """Заглушка: позволяет коду с метриками работать без prometheus_client."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def labels(self, *args: Any, **kwargs: Any) -> "_NullMetric":
            return self

        def inc(self, *args: Any, **kwargs: Any) -> None:
            pass

        def observe(self, *args: Any, **kwargs: Any) -> None:
            pass

        def set(self, *args: Any, **kwargs: Any) -> None:
            pass

    Counter = Gauge = Histogram = _NullMetric  # type: ignore[assignment,misc]

    def generate_latest(*args: Any, **kwargs: Any) -> bytes:  # type: ignore[misc]
        return b"# prometheus_client not installed\n"


def _metric(cls: Any, name: str, documentation: str, **kwargs: Any) -> Any:
    """Создаёт метрику, но не падает при повторном импорте модуля.

    В ноутбуках ячейку с метриками часто запускают дважды, и prometheus_client
    бросает ValueError: Duplicated timeseries. Здесь мы переиспользуем уже
    зарегистрированный коллектор.
    """
    try:
        return cls(name, documentation, **kwargs)
    except ValueError:
        if REGISTRY is None:  # pragma: no cover
            return cls(name, documentation, **kwargs)
        # приватный индекс prometheus_client: другого публичного способа нет
        existing = getattr(REGISTRY, "_names_to_collectors", {})
        for key in (name, f"{name}_total", name.removesuffix("_total")):
            if key in existing:
                return existing[key]
        raise


LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 90.0)
SCORE_BUCKETS = (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

# ВАЖНО про метки (labels): никогда не кладите в них user_id, текст вопроса,
# doc_id или trace_id. Каждое уникальное значение метки — отдельный временной ряд,
# это убивает Prometheus (проблема cardinality explosion).
RAG_REQUESTS = _metric(
    Counter,
    "rag_requests_total",
    "Запросы к RAG-пайплайну",
    labelnames=("outcome", "tenant"),
)
RAG_STAGE_SECONDS = _metric(
    Histogram,
    "rag_stage_duration_seconds",
    "Длительность стадии пайплайна",
    labelnames=("stage",),
    buckets=LATENCY_BUCKETS,
)
RAG_CHUNKS_USED = _metric(
    Histogram,
    "rag_chunks_used",
    "Сколько чанков попало в промпт",
    buckets=(0, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20),
)
RAG_TOP_SIMILARITY = _metric(
    Histogram,
    "rag_top_similarity",
    "Косинусная близость лучшего найденного чанка",
    buckets=SCORE_BUCKETS,
)
RAG_NO_CONTEXT = _metric(
    Counter,
    "rag_no_context_total",
    "Запросы, для которых не нашлось контекста выше порога",
)
RAG_ACL_DENIED = _metric(
    Counter,
    "rag_acl_denied_total",
    "Чанки, отфильтрованные по правам доступа (защитный пост-контроль)",
)
RAG_INJECTION_BLOCKED = _metric(
    Counter,
    "rag_injection_blocked_total",
    "Срабатывания детектора prompt injection",
    labelnames=("stage", "rule"),
)
RAG_PII_MASKED = _metric(
    Counter,
    "rag_pii_masked_total",
    "Замаскированные сущности ПДн",
    labelnames=("entity",),
)
RAG_ANSWER_BLOCKED = _metric(
    Counter,
    "rag_answer_blocked_total",
    "Ответы, заблокированные выходным фильтром",
    labelnames=("reason",),
)
RAG_LLM_TOKENS = _metric(
    Counter,
    "rag_llm_tokens_total",
    "Токены LLM",
    labelnames=("kind",),
)
RAG_INDEX_CHUNKS = _metric(
    Gauge,
    "rag_index_chunks",
    "Число чанков в индексе",
)
RAG_INDEX_LAST_SUCCESS = _metric(
    Gauge,
    "rag_index_last_success_timestamp_seconds",
    "Unix-время последней успешной переиндексации",
)

# --------------------------------------------------------------------------
# Структурные логи
# --------------------------------------------------------------------------

_trace_id: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="-")

_STD_LOG_FIELDS = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
    "taskName",
    "message",
    "asctime",
}


class JsonFormatter(logging.Formatter):
    """Пишет логи одной JSON-строкой на событие.

    Такой формат парсится Loki/ELK/Datadog без regex-магии. Человеку читать
    сложнее — для локальной разработки оставьте обычный текстовый формат.
    """

    def __init__(self, service: str = "rag") -> None:
        super().__init__()
        self.service = service

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "service": self.service,
            "logger": record.name,
            "event": record.getMessage(),
            "trace_id": getattr(record, "trace_id", None) or _trace_id.get(),
        }
        for key, value in record.__dict__.items():
            if key not in _STD_LOG_FIELDS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(
    level: str = "INFO", json_logs: bool | None = None, service: str = "rag"
) -> logging.Logger:
    """Настраивает корневой логгер один раз. Возвращает логгер сервиса."""
    if json_logs is None:
        json_logs = os.getenv("RAG_JSON_LOGS", "1") not in {"0", "false", "no"}

    root = logging.getLogger()
    root.setLevel(level.upper())
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(stream=sys.stdout)
    if json_logs:
        handler.setFormatter(JsonFormatter(service=service))
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s")
        )
    root.addHandler(handler)
    # uvicorn любит дублировать логи — отдаём их корневому обработчику
    for noisy in ("uvicorn.access", "uvicorn.error", "httpx"):
        logging.getLogger(noisy).handlers = []
        logging.getLogger(noisy).propagate = True
    logging.getLogger("httpx").setLevel("WARNING")
    return logging.getLogger(service)


log = logging.getLogger("rag")


def new_trace_id() -> str:
    """Новый идентификатор запроса (короткий, чтобы удобно копировать из логов)."""
    trace_id = uuid.uuid4().hex[:16]
    _trace_id.set(trace_id)
    return trace_id


def set_trace_id(trace_id: str) -> None:
    _trace_id.set(trace_id)


def get_trace_id() -> str:
    return _trace_id.get()


def log_event(event: str, level: int = logging.INFO, **fields: Any) -> None:
    """Логирует событие со структурными полями (без f-строк в сообщении)."""
    log.log(level, event, extra={"trace_id": _trace_id.get(), **fields})


# --------------------------------------------------------------------------
# Трейсинг стадий
# --------------------------------------------------------------------------


class Span:
    """Одна стадия пайплайна: время + произвольные атрибуты."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.attributes: dict[str, Any] = {}
        self.duration_s: float = 0.0

    def set(self, **fields: Any) -> "Span":
        self.attributes.update(fields)
        return self


@contextmanager
def stage(name: str, **fields: Any) -> Iterator[Span]:
    """Замеряет стадию, пишет лог и гистограмму.

    Использование:
        with stage("retrieve", k=20) as span:
            hits = search(...)
            span.set(hits=len(hits))
    """
    span = Span(name)
    span.attributes.update(fields)
    started = time.perf_counter()
    try:
        yield span
    except Exception as exc:  # noqa: BLE001 - логируем и пробрасываем
        span.duration_s = time.perf_counter() - started
        RAG_STAGE_SECONDS.labels(stage=name).observe(span.duration_s)
        log_event(
            "stage.error",
            level=logging.ERROR,
            stage=name,
            duration_ms=round(span.duration_s * 1000, 1),
            error=type(exc).__name__,
            error_message=str(exc)[:200],
            **span.attributes,
        )
        raise
    else:
        span.duration_s = time.perf_counter() - started
        RAG_STAGE_SECONDS.labels(stage=name).observe(span.duration_s)
        log_event(
            "stage.done",
            stage=name,
            duration_ms=round(span.duration_s * 1000, 1),
            **span.attributes,
        )


def metrics_snapshot() -> str:
    """Текущее состояние метрик в текстовом формате Prometheus (для отладки)."""
    return generate_latest().decode("utf-8")


def find_metric_lines(substring: str) -> list[str]:
    """Удобно в ноутбуке: показать только интересующие строки /metrics."""
    return [
        line
        for line in metrics_snapshot().splitlines()
        if substring in line and not line.startswith("#")
    ]


__all__ = [
    "configure_logging",
    "JsonFormatter",
    "log_event",
    "new_trace_id",
    "set_trace_id",
    "get_trace_id",
    "stage",
    "Span",
    "metrics_snapshot",
    "find_metric_lines",
    "PROMETHEUS_AVAILABLE",
    "CONTENT_TYPE_LATEST",
    "RAG_REQUESTS",
    "RAG_STAGE_SECONDS",
    "RAG_CHUNKS_USED",
    "RAG_TOP_SIMILARITY",
    "RAG_NO_CONTEXT",
    "RAG_ACL_DENIED",
    "RAG_INJECTION_BLOCKED",
    "RAG_PII_MASKED",
    "RAG_ANSWER_BLOCKED",
    "RAG_LLM_TOKENS",
    "RAG_INDEX_CHUNKS",
    "RAG_INDEX_LAST_SUCCESS",
]
