"""Структурные логи (урок 3.7): JSONL + request_id через contextvars.

Два правила из урока:

1. request_id привязывается ОДИН раз на границе запроса
   (bind_contextvars в /ask) - и каждое событие глубже по стеку получает
   его автоматически, без протаскивания через аргументы.
2. Текст вопроса в лог не попадает НИКОГДА - только отпечаток (длина +
   хэш): логи живут дольше и читаются шире, чем сами диалоги, и ПДн
   в них не место. Хэша достаточно, чтобы сгруппировать повторы.

Конфигурация принимает ПОТОК, а не путь: прод открывает файл из настроек,
тесты подставляют io.StringIO и читают события как данные.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import IO

import structlog


def question_fingerprint(question: str) -> dict:
    """ПДн-безопасное описание вопроса: длина и хэш вместо текста."""
    return {
        "question_len": len(question),
        "question_hash": hashlib.sha256(
            question.strip().lower().encode()).hexdigest()[:12],
    }


def open_log_file(path: str | Path) -> IO[str]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return open(path, "a", encoding="utf-8")


def configure_logging(stream: IO[str]) -> None:
    """Прод-конфигурация structlog: JSON-строки, по событию на строку."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(ensure_ascii=False),
        ],
        logger_factory=structlog.WriteLoggerFactory(file=stream),
        # без кэша: переконфигурация (например, в тестах) подхватывается
        # всеми уже созданными логгерами
        cache_logger_on_first_use=False,
    )
