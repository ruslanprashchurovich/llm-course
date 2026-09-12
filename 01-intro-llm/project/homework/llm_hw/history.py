"""Персистентная история чата — ЗАДАНИЕ 4.

История диалога хранится в ~/.llm_history.json и переживает перезапуск утилиты.
Ключевой нюанс из задания: обрезку `MAX_HISTORY` применяем **при загрузке** —
файл со временем распухает, а в модель нельзя тащить бесконечный контекст (урок 1.1).
Память диалога живёт в промпте, а не в модели (урок 1.6), поэтому «сохранить чат» =
сохранить список сообщений, а «забыть» = удалить файл.
"""

from __future__ import annotations

import json
from pathlib import Path

HISTORY_PATH = Path.home() / ".llm_history.json"
MAX_HISTORY = 12  # держим последние N сообщений — та же обрезка, что в базовом проекте


def load_history() -> list[dict]:
    """Читает историю; при отсутствии/порче файла молча возвращает [].

    Обрезаем до MAX_HISTORY СРАЗУ при загрузке (см. docstring модуля).
    """
    try:
        data = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return []
    if not isinstance(data, list):
        return []
    messages = [
        m for m in data
        if isinstance(m, dict) and m.get("role") in ("user", "assistant")
    ]
    return messages[-MAX_HISTORY:]


def save_history(history: list[dict]) -> None:
    """Пишет историю, обрезав хвост до MAX_HISTORY — файл не растёт бесконечно."""
    HISTORY_PATH.write_text(
        json.dumps(history[-MAX_HISTORY:], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def clear_history() -> None:
    """Забыть контекст = удалить файл истории (/clear и --fresh)."""
    HISTORY_PATH.unlink(missing_ok=True)
