"""GET /models — инвентарь диска Ollama (практика 5.7, базовый уровень)."""

from app.config import Settings
from app.ollama_client import OllamaUnavailable
from fastapi.testclient import TestClient
from tests.conftest import FakeOllama

AUTH = {"X-API-Token": "secret"}


def test_models_lists_disk_models_with_sizes_digests_and_roles(make_app, fake):
    settings = Settings(api_tokens=("secret",), fallback_model="qwen2.5:1.5b")
    with TestClient(make_app(fake, settings)) as client:
        resp = client.get("/models", headers=AUTH)
    assert resp.status_code == 200
    body = resp.json()
    assert body["primary"] == "qwen2.5:3b"
    assert body["fallback"] == "qwen2.5:1.5b"
    assert body["primary_installed"] is True

    by_name = {m["name"]: m for m in body["models"]}
    primary = by_name["qwen2.5:3b"]
    assert primary["role"] == "primary"
    assert primary["size_bytes"] == 1_929_912_432
    assert primary["size_gb"] == 1.8            # computed-поле: байты -> ГиБ с округлением
    assert primary["digest"].startswith("sha256:357c53fb659c")
    assert primary["quantization"] == "Q4_K_M"
    assert by_name["qwen2.5:1.5b"]["role"] == "fallback"
    assert fake.calls["tags"] == 1


def test_models_requires_token(make_app, fake):
    with TestClient(make_app(fake)) as client:
        assert client.get("/models").status_code == 401
    assert fake.calls["tags"] == 0  # до Ollama не дошли


def test_models_502_when_ollama_down(make_app):
    fake = FakeOllama(tags_error=OllamaUnavailable("connect refused"))
    with TestClient(make_app(fake)) as client:
        resp = client.get("/models", headers=AUTH)
    assert resp.status_code == 502
    assert "недоступна" in resp.json()["detail"]


def test_models_reports_missing_primary(make_app):
    fake = FakeOllama(tags=[])  # диск пуст — ровно как после ollama rm
    with TestClient(make_app(fake)) as client:
        body = client.get("/models", headers=AUTH).json()
    assert body["models"] == []
    assert body["primary_installed"] is False
