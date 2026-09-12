"""Retry-матрица OllamaClient — через httpx.MockTransport, без сети и без Ollama."""

import asyncio
import json

import httpx
import pytest

from app.config import Settings
from app.ollama_client import (OllamaBadRequest, OllamaBusy, OllamaClient,
                               OllamaUnavailable)

FAST = Settings(retries=2, backoff_base_s=0.01)

OK_JSON = {"response": "ok", "eval_count": 5, "eval_duration": 1_000_000_000}


def make_client(handler) -> OllamaClient:
    return OllamaClient(FAST, transport=httpx.MockTransport(handler))


def test_retry_on_503_then_success():
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(503) if attempts["n"] < 3 else httpx.Response(200, json=OK_JSON)

    data = asyncio.run(make_client(handler).generate("hi", {"num_predict": 5}))
    assert data["response"] == "ok"
    assert attempts["n"] == 3  # два повтора по матрице — и успех


def test_busy_after_all_retries():
    attempts = {"n": 0}

    def handler(request):
        attempts["n"] += 1
        return httpx.Response(503)

    with pytest.raises(OllamaBusy):
        asyncio.run(make_client(handler).generate("hi", {"num_predict": 5}))
    assert attempts["n"] == FAST.retries + 1


def test_connect_error_becomes_unavailable():
    attempts = {"n": 0}

    def handler(request):
        attempts["n"] += 1
        raise httpx.ConnectError("порт закрыт", request=request)

    with pytest.raises(OllamaUnavailable):
        asyncio.run(make_client(handler).generate("hi", {"num_predict": 5}))
    assert attempts["n"] == FAST.retries + 1


def test_no_retry_on_4xx():
    attempts = {"n": 0}

    def handler(request):
        attempts["n"] += 1
        return httpx.Response(404, text='{"error":"model not found"}')

    with pytest.raises(OllamaBadRequest):
        asyncio.run(make_client(handler).generate("hi", {"num_predict": 5}))
    assert attempts["n"] == 1  # конфиг повтором не чинится (матрица 5.3)


def test_no_retry_on_read_timeout():
    attempts = {"n": 0}

    def handler(request):
        attempts["n"] += 1
        raise httpx.ReadTimeout("долгая генерация", request=request)

    with pytest.raises(httpx.ReadTimeout):
        asyncio.run(make_client(handler).generate("hi", {"num_predict": 5}))
    assert attempts["n"] == 1  # ReadTimeout летит наружу БЕЗ повторов


def test_think_is_a_top_level_request_field(monkeypatch):
    """think — поле запроса Ollama, не options (урок 5.2); по умолчанию False для всех моделей."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json=OK_JSON)

    asyncio.run(make_client(handler).generate("hi", {"num_predict": 5}))
    assert seen["think"] is False and "think" not in seen["options"]

    monkeypatch.setenv("PROXY_THINK", "1")
    assert Settings.from_env().think is True
    monkeypatch.delenv("PROXY_THINK")
    assert Settings.from_env().think is False
