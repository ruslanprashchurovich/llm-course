"""Хранение истории вопросов и ответов в SQLite (через aiosqlite).

Для однопользовательского внутреннего сервиса SQLite достаточно: нет
отдельного процесса, нет сетевых задержек, файл легко бэкапится.
Компромиссы и путь миграции на PostgreSQL описаны в ADR-0004 и уроке 6.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import aiosqlite

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    sources TEXT NOT NULL,
    model TEXT NOT NULL,
    took_ms INTEGER NOT NULL
);
"""


class HistoryRepository:
    """Репозиторий истории диалогов."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path

    async def init(self) -> None:
        """Создаёт файл БД и таблицу, если их ещё нет (идемпотентно)."""
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(_SCHEMA)
            await db.commit()
        logger.info("История: SQLite готова (%s)", self._db_path)

    async def save(
        self,
        *,
        question: str,
        answer: str,
        sources: list[str],
        model: str,
        took_ms: int,
    ) -> None:
        """Сохраняет одну пару вопрос-ответ.

        ``sources`` сериализуем в JSON-строку: список имён файлов маленький,
        и отдельная таблица связей для v1 — лишняя сложность (см. урок 6).
        """
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "INSERT INTO interactions (question, answer, sources, model, took_ms) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    question,
                    answer,
                    json.dumps(sources, ensure_ascii=False),
                    model,
                    took_ms,
                ),
            )
            await db.commit()

    async def list_recent(self, limit: int = 20) -> list[dict[str, Any]]:
        """Последние записи, новые первыми."""
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT id, created_at, question, answer, sources, model, took_ms "
                "FROM interactions ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            rows = await cursor.fetchall()
        return [
            {
                "id": row["id"],
                "created_at": row["created_at"],
                "question": row["question"],
                "answer": row["answer"],
                "sources": json.loads(row["sources"]),
                "model": row["model"],
                "took_ms": row["took_ms"],
            }
            for row in rows
        ]
