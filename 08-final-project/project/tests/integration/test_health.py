"""Интеграционные тесты служебных эндпоинтов: /health, /ready, /metrics."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from tests.conftest import FakeLLM, FakeVectorStore


def test_health_always_ok(bare_client: TestClient) -> None:
    response = bare_client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"]


def test_ready_degraded_without_services(bare_client: TestClient) -> None:
    response = bare_client.get("/ready")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["checks"][0]["ok"] is False
    assert body["failed"] == ["services"]


@pytest.fixture()
def ready_client(test_settings: Settings) -> Iterator[TestClient]:
    app = create_app(test_settings)
    app.state.vectorstore = FakeVectorStore(is_healthy=True)
    app.state.llm = FakeLLM(is_healthy=True)
    with TestClient(app) as client:
        yield client


def test_ready_ok_with_healthy_dependencies(ready_client: TestClient) -> None:
    response = ready_client.get("/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert {check["name"] for check in body["checks"]} == {"qdrant", "ollama"}
    assert body["failed"] == []


def test_ready_degraded_when_ollama_down(test_settings: Settings) -> None:
    app = create_app(test_settings)
    app.state.vectorstore = FakeVectorStore(is_healthy=True)
    app.state.llm = FakeLLM(is_healthy=False)
    with TestClient(app) as client:
        response = client.get("/ready")
    assert response.status_code == 503
    checks = {check["name"]: check["ok"] for check in response.json()["checks"]}
    assert checks == {"qdrant": True, "ollama": False}
    # computed-поле схемы: виновники одним списком, без обхода checks на клиенте.
    assert response.json()["failed"] == ["ollama"]


def test_not_found_uses_error_schema(bare_client: TestClient) -> None:
    """Даже 404 от роутера приходит в едином формате ErrorResponse."""
    response = bare_client.get("/nope", headers={"X-Request-ID": "req-404"})
    assert response.status_code == 404
    assert response.json() == {"detail": "Not Found", "request_id": "req-404"}


def test_metrics_exposed(bare_client: TestClient) -> None:
    bare_client.get("/health")  # хотя бы один запрос, чтобы счётчик появился
    response = bare_client.get("/metrics")
    assert response.status_code == 200
    assert "http_requests_total" in response.text


def test_request_id_header(bare_client: TestClient) -> None:
    response = bare_client.get("/health")
    assert response.headers.get("X-Request-ID")
    # Переданный клиентом request-id должен вернуться без изменений.
    response = bare_client.get("/health", headers={"X-Request-ID": "abc123"})
    assert response.headers["X-Request-ID"] == "abc123"
