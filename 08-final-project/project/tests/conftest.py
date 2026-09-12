"""Общие фикстуры: приложение с фейковыми сервисами вместо тяжёлых зависимостей.

Ключевая идея: тесты НЕ должны требовать GPU, скачанных моделей, Qdrant
или Ollama. Мы подменяем сервисы на границе (app.state) фейками с тем же
интерфейсом — как в обычном backend'е мокают платёжный шлюз или SMTP.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

# Страховка: даже если кто-то создаст Settings() без аргументов,
# окружение останется тестовым и lifespan не полезет качать модели.
os.environ.setdefault("APP_ENVIRONMENT", "test")

from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402
from app.services.rag import RankedChunk  # noqa: E402
from app.services.vectorstore import ScoredChunk  # noqa: E402

CHUNK = ScoredChunk(
    text="Деплой на прод выполняется по тегу vX.Y.Z: пайплайн GitLab CI запускает job deploy-prod.",
    source="05-deploy-guide.md",
    title="Гид по деплою",
    chunk_index=0,
    score=0.87,
)

STREAM_TOKENS = ["Деплой ", "запускается ", "по ", "тегу ", "[1]."]


class FakeRag:
    """Фейковый RAG-сервис: без моделей и сети, отвечает мгновенно и предсказуемо."""

    model_name = "fake-model"

    def __init__(self) -> None:
        self.search_calls: list[dict[str, Any]] = []
        self.answer_calls: list[str] = []

    async def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        fetch_k: int | None = None,
        use_reranker: bool | None = None,
        source: str | None = None,
    ) -> list[RankedChunk]:
        self.search_calls.append(
            {
                "query": query,
                "top_k": top_k,
                "fetch_k": fetch_k,
                "use_reranker": use_reranker,
                "source": source,
            }
        )
        return [RankedChunk(chunk=CHUNK, rerank_score=0.95)]

    async def answer(
        self,
        question: str,
        *,
        top_k: int | None = None,
        temperature: float | None = None,
    ) -> tuple[str, list[ScoredChunk]]:
        self.answer_calls.append(question)
        return "Деплой запускается по тегу [1].", [CHUNK]

    async def stream_answer(
        self,
        question: str,
        *,
        top_k: int | None = None,
        temperature: float | None = None,
    ) -> tuple[list[ScoredChunk], AsyncIterator[str]]:
        async def _tokens() -> AsyncIterator[str]:
            for token in STREAM_TOKENS:
                yield token

        return [CHUNK], _tokens()


class FakeHistory:
    """Фейковая история: пишет в память, чтобы тесты могли проверить вызовы."""

    def __init__(self) -> None:
        self.saved: list[dict[str, Any]] = []

    async def save(self, **kwargs: Any) -> None:
        self.saved.append(kwargs)

    async def list_recent(self, limit: int = 20) -> list[dict[str, Any]]:
        items = [
            {
                "id": 1,
                "created_at": "2026-01-01 00:00:00",
                "question": "Как деплоить?",
                "answer": "По тегу [1].",
                "sources": ["05-deploy-guide.md"],
                "model": "fake-model",
                "took_ms": 5,
            }
        ]
        return items[:limit]


class FakeVectorStore:
    """Фейковый Qdrant для проверок /ready."""

    def __init__(self, is_healthy: bool = True) -> None:
        self._is_healthy = is_healthy

    async def healthy(self) -> bool:
        return self._is_healthy


class FakeLLM:
    """Фейковая Ollama для проверок /ready."""

    model = "fake-model"

    def __init__(self, is_healthy: bool = True) -> None:
        self._is_healthy = is_healthy

    async def healthy(self) -> bool:
        return self._is_healthy


@pytest.fixture()
def test_settings(tmp_path: Any) -> Settings:
    """Настройки для тестов: environment=test отключает тяжёлый lifespan."""
    return Settings(
        _env_file=None,  # игнорируем локальный .env разработчика
        environment="test",
        history_db_path=str(tmp_path / "history.db"),
    )


@pytest.fixture()
def fake_rag() -> FakeRag:
    return FakeRag()


@pytest.fixture()
def fake_history() -> FakeHistory:
    return FakeHistory()


@pytest.fixture()
def client(
    test_settings: Settings,
    fake_rag: FakeRag,
    fake_history: FakeHistory,
) -> Iterator[TestClient]:
    """HTTP-клиент к приложению с фейковыми сервисами.

    TestClient обязательно используем как контекстный менеджер —
    иначе lifespan (startup/shutdown) не выполнится.
    """
    app = create_app(test_settings)
    app.state.rag = fake_rag
    app.state.history = fake_history
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def bare_client(test_settings: Settings) -> Iterator[TestClient]:
    """Клиент к приложению БЕЗ сервисов — для тестов деградации (503)."""
    app = create_app(test_settings)
    with TestClient(app) as test_client:
        yield test_client
