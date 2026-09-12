"""FastAPI-зависимости: доступ к сервисам из ``app.state``.

Сервисы создаются один раз в lifespan (app/main.py) и складываются
в ``app.state``. Зависимости достают их оттуда — а в тестах мы просто
кладём в state фейковые реализации (см. tests/conftest.py).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import HTTPException, Request, status

if TYPE_CHECKING:
    from app.services.history import HistoryRepository
    from app.services.rag import RagService


def _require(request: Request, name: str) -> Any:
    """Возвращает сервис из app.state или отвечает 503, если его нет.

    503 честнее, чем загадочный AttributeError: клиент видит, что сервис
    жив, но ещё/уже не готов обслуживать запросы.
    """
    service = getattr(request.app.state, name, None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Сервис '{name}' не инициализирован",
        )
    return service


def get_rag(request: Request) -> RagService:
    """RAG-оркестратор (поиск + генерация)."""
    return _require(request, "rag")


def get_history(request: Request) -> HistoryRepository:
    """Репозиторий истории диалогов."""
    return _require(request, "history")
