"""Unit-тесты SQLite-репозитория истории (реальный файл во временной папке)."""

from __future__ import annotations

import asyncio
from pathlib import Path

from app.services.history import HistoryRepository


def test_history_roundtrip(tmp_path: Path) -> None:
    async def scenario() -> list[dict]:
        repo = HistoryRepository(str(tmp_path / "history.db"))
        await repo.init()
        await repo.init()  # повторная инициализация не должна падать
        await repo.save(
            question="Как деплоить?",
            answer="По тегу [1].",
            sources=["05-deploy-guide.md"],
            model="fake-model",
            took_ms=1234,
        )
        await repo.save(
            question="Где секреты?",
            answer="В Vault [1].",
            sources=["10-secrets-management.md"],
            model="fake-model",
            took_ms=999,
        )
        return await repo.list_recent(limit=10)

    items = asyncio.run(scenario())
    assert len(items) == 2
    # Новые записи первыми.
    assert items[0]["question"] == "Где секреты?"
    assert items[1]["question"] == "Как деплоить?"
    assert items[1]["sources"] == ["05-deploy-guide.md"]
    assert items[1]["took_ms"] == 1234


def test_list_recent_respects_limit(tmp_path: Path) -> None:
    async def scenario() -> list[dict]:
        repo = HistoryRepository(str(tmp_path / "history.db"))
        await repo.init()
        for i in range(5):
            await repo.save(
                question=f"q{i}", answer=f"a{i}", sources=[], model="m", took_ms=i
            )
        return await repo.list_recent(limit=2)

    items = asyncio.run(scenario())
    assert len(items) == 2
    assert items[0]["question"] == "q4"
