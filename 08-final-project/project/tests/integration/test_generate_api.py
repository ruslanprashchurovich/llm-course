"""Интеграционные тесты POST /api/generate: JSON, SSE-стриминг, история, ошибки."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from app.config import Settings
from app.main import create_app
from app.schemas import DoneEvent, SourcesEvent, TokenEvent, parse_sse
from app.services.llm import LLMError
from fastapi.testclient import TestClient

from tests.conftest import STREAM_TOKENS, FakeHistory, FakeRag


def test_generate_json_answer(client: TestClient, fake_history: FakeHistory) -> None:
    response = client.post("/api/generate", json={"query": "Как деплоить на прод?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Деплой запускается по тегу [1]."
    assert body["model"] == "fake-model"
    assert isinstance(body["took_ms"], int)
    # Источники пронумерованы так же, как в промпте: [1], [2], ...
    assert body["sources"][0]["number"] == 1
    assert body["sources"][0]["source"] == "05-deploy-guide.md"
    # Производные поля схемы: что процитировано, нет ли выдуманных номеров, отказ ли это.
    assert body["cited"] == [1]
    assert body["dangling_citations"] == []
    assert body["is_refusal"] is False

    # Ответ сохранён в историю - вместе с источниками и моделью.
    assert len(fake_history.saved) == 1
    saved = fake_history.saved[0]
    assert saved["question"] == "Как деплоить на прод?"
    assert saved["sources"] == ["05-deploy-guide.md"]
    assert saved["model"] == "fake-model"


def test_generate_stream_sse(client: TestClient, fake_history: FakeHistory) -> None:
    with client.stream(
        "POST", "/api/generate", json={"query": "Как деплоить?", "stream": True}
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        lines = [line for line in response.iter_lines() if line]

    events = [line.split(" ", 1)[1] for line in lines if line.startswith("event:")]
    # Контракт стрима: сначала sources, затем токены, в конце done.
    assert events[0] == "sources"
    assert events[-1] == "done"
    assert events[1:-1] == ["token"] * len(STREAM_TOKENS)

    # Каждое событие разбирается в типизированную модель (discriminated union).
    datas = [line.split(" ", 1)[1] for line in lines if line.startswith("data:")]
    typed = [parse_sse(name, data) for name, data in zip(events, datas, strict=True)]
    assert isinstance(typed[0], SourcesEvent) and typed[0].sources[0].number == 1
    assert all(isinstance(event, TokenEvent) for event in typed[1:-1])
    assert isinstance(typed[-1], DoneEvent) and typed[-1].model == "fake-model"

    # Полный ответ (склейка токенов) сохранён в историю даже в стриминге.
    assert len(fake_history.saved) == 1
    assert fake_history.saved[0]["answer"] == "".join(STREAM_TOKENS)


def test_generate_passes_parameters(client: TestClient, fake_rag: FakeRag) -> None:
    client.post(
        "/api/generate", json={"query": "вопрос", "top_k": 2, "temperature": 0.0}
    )
    assert fake_rag.answer_calls == ["вопрос"]


def test_generate_validation(client: TestClient, fake_rag: FakeRag) -> None:
    bad_bodies: list[dict[str, Any]] = [
        {"query": ""},
        {"query": "x", "temperature": 5.0},
        {"query": "x", "stream": "yes"},  # strict: булево строкой — не булево
        {"query": "x", "top_k": "3"},  # strict: число строкой — не число
        {"query": "x", "steam": True},  # extra=forbid: опечатка не проходит молча
    ]
    for body in bad_bodies:
        assert client.post("/api/generate", json=body).status_code == 422, body
    assert fake_rag.answer_calls == []  # до RagService мусор не дошёл


def test_history_endpoint(client: TestClient) -> None:
    response = client.get("/api/history?limit=5")
    assert response.status_code == 200
    items = response.json()
    assert items[0]["question"] == "Как деплоить?"
    assert items[0]["sources"] == ["05-deploy-guide.md"]
    # Строка SQLite «2026-01-01 00:00:00» -> ISO 8601 с явным UTC.
    assert items[0]["created_at"] == "2026-01-01T00:00:00Z"


def test_generate_unavailable_without_services(bare_client: TestClient) -> None:
    response = bare_client.post("/api/generate", json={"query": "x"})
    assert response.status_code == 503
    assert set(response.json()) == {"detail", "request_id"}


class FailingRag(FakeRag):
    """RAG, у которого «упала» LLM: проверяем формат ответа 502."""

    async def answer(self, question: str, **kwargs: Any) -> tuple[str, list[Any]]:
        raise LLMError("Ollama не отвечает")


@pytest.fixture()
def failing_client(
    test_settings: Settings, fake_history: FakeHistory
) -> Iterator[TestClient]:
    app = create_app(test_settings)
    app.state.rag = FailingRag()
    app.state.history = fake_history
    with TestClient(app) as test_client:
        yield test_client


def test_generate_llm_error_uses_error_schema(
    failing_client: TestClient, fake_history: FakeHistory
) -> None:
    response = failing_client.post(
        "/api/generate", json={"query": "x"}, headers={"X-Request-ID": "req-7"}
    )
    assert response.status_code == 502
    body = response.json()
    assert body["detail"].startswith("Ошибка LLM-сервера")
    assert body["request_id"] == "req-7"
    assert fake_history.saved == []  # упавший ответ в историю не пишем
