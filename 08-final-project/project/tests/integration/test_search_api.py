"""Интеграционные тесты POST /api/search (RAG-сервис замокан)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from tests.conftest import FakeRag


def test_search_returns_results(client: TestClient) -> None:
    response = client.post("/api/search", json={"query": "как деплоить на прод?"})
    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "как деплоить на прод?"
    assert isinstance(body["took_ms"], int)
    assert len(body["results"]) == 1
    # computed-поля схемы: сколько нашли и по какой шкале отсортировано.
    assert body["count"] == 1
    assert body["ranked_by"] == "rerank_score"

    result = body["results"][0]
    assert result["source"] == "05-deploy-guide.md"
    assert result["title"] == "Гид по деплою"
    assert result["score"] == 0.87
    assert result["rerank_score"] == 0.95


def test_search_passes_parameters_to_service(client: TestClient, fake_rag: FakeRag) -> None:
    client.post(
        "/api/search",
        json={
            "query": "деплой",
            "top_k": 3,
            "fetch_k": 30,
            "use_reranker": True,
            "source": "05-deploy-guide.md",
        },
    )
    assert fake_rag.search_calls == [
        {
            "query": "деплой",
            "top_k": 3,
            "fetch_k": 30,
            "use_reranker": True,
            "source": "05-deploy-guide.md",
        }
    ]


def test_search_empty_query_rejected(client: TestClient) -> None:
    assert client.post("/api/search", json={"query": ""}).status_code == 422
    assert client.post("/api/search", json={"query": "   "}).status_code == 422


def test_search_top_k_out_of_range_rejected(client: TestClient) -> None:
    response = client.post("/api/search", json={"query": "x", "top_k": 100})
    assert response.status_code == 422


def test_search_garbage_rejected_before_service(client: TestClient, fake_rag: FakeRag) -> None:
    """Строгая схема: опечатка, число строкой, путь вместо имени — 422 и НЕ-вызов RAG."""
    garbage = [
        {"query": "x", "topk": 3},  # опечатка в имени поля
        {"query": "x", "top_k": "3"},  # число строкой
        {"query": "x", "source": "../etc/passwd"},  # путь вместо имени файла
    ]
    for payload in garbage:
        assert client.post("/api/search", json=payload).status_code == 422, payload
    assert fake_rag.search_calls == []


def test_search_fetch_k_rules(client: TestClient) -> None:
    response = client.post("/api/search", json={"query": "x", "top_k": 5, "fetch_k": 3})
    assert response.status_code == 422
    assert "fetch_k" in response.json()["detail"][0]["msg"]
    response = client.post(
        "/api/search", json={"query": "x", "fetch_k": 30, "use_reranker": False}
    )
    assert response.status_code == 422


def test_search_unavailable_without_services(bare_client: TestClient) -> None:
    response = bare_client.post(
        "/api/search", json={"query": "x"}, headers={"X-Request-ID": "req-42"}
    )
    assert response.status_code == 503
    body = response.json()
    assert "не инициализирован" in body["detail"]
    # Единый формат ошибок: request_id в теле совпадает с заголовком.
    assert body["request_id"] == "req-42" == response.headers["X-Request-ID"]
