"""Тесты эндпоинтов на фейках. TestClient с `with` — чтобы отработал lifespan."""

import httpx
from app.config import Settings
from app.ollama_client import OllamaBusy, OllamaUnavailable
from fastapi.testclient import TestClient
from tests.conftest import TEST_SETTINGS, FakeOllama

AUTH = {"X-API-Token": "secret"}


# ---------------------------------------------------------------- auth
def test_generate_requires_token(make_app, fake):
    with TestClient(make_app(fake)) as client:
        assert client.post("/generate", json={"prompt": "hi"}).status_code == 401
        bad = client.post(
            "/generate", json={"prompt": "hi"}, headers={"X-API-Token": "wrong"}
        )
        assert bad.status_code == 401
    assert fake.calls["generate"] == 0  # до Ollama запрос не дошёл


def test_healthz_open_without_token(make_app, fake):
    with TestClient(make_app(fake)) as client:
        assert client.get("/healthz").status_code == 200


# ---------------------------------------------------------------- happy path
def test_generate_ok(make_app, fake):
    with TestClient(make_app(fake)) as client:
        resp = client.post("/generate", json={"prompt": "привет"}, headers=AUTH)
    assert resp.status_code == 200
    body = resp.json()
    assert body["text"].startswith("эхо:")
    assert body["completion_tokens"] == 40
    assert body["tokens_per_second"] == 20.0  # 40 токенов за 2 секунды
    assert fake.calls["generate"] == 1


def test_num_predict_cap_enforced(make_app, fake):
    with TestClient(make_app(fake)) as client:
        client.post(
            "/generate", json={"prompt": "hi", "num_predict": 100000}, headers=AUTH
        )
    assert fake.last_options["num_predict"] == TEST_SETTINGS.num_predict_cap

    with TestClient(make_app(fake)) as client:  # без num_predict — тоже потолок
        client.post("/generate", json={"prompt": "hi"}, headers=AUTH)
    assert fake.last_options["num_predict"] == TEST_SETTINGS.num_predict_cap


def test_validation_rejects_empty_prompt(make_app, fake):
    with TestClient(make_app(fake)) as client:
        resp = client.post("/generate", json={"prompt": ""}, headers=AUTH)
    assert resp.status_code == 422
    assert fake.calls["generate"] == 0


# ---------------------------------------------------------------- ошибки апстрима
def test_read_timeout_maps_to_504_without_retry(make_app):
    fake = FakeOllama(error=httpx.ReadTimeout("генерация слишком долгая"))
    with TestClient(make_app(fake)) as client:
        resp = client.post("/generate", json={"prompt": "hi"}, headers=AUTH)
    assert resp.status_code == 504
    assert fake.calls["generate"] == 1  # НЕ-вызов: повторов не было (матрица 5.3)


def test_unavailable_maps_to_502(make_app):
    fake = FakeOllama(error=OllamaUnavailable("connect refused"))
    with TestClient(make_app(fake)) as client:
        resp = client.post("/generate", json={"prompt": "hi"}, headers=AUTH)
    assert resp.status_code == 502


def test_busy_maps_to_503(make_app):
    fake = FakeOllama(error=OllamaBusy("HTTP 503"))
    with TestClient(make_app(fake)) as client:
        resp = client.post("/generate", json={"prompt": "hi"}, headers=AUTH)
    assert resp.status_code == 503


# ---------------------------------------------------------------- healthz
def test_healthz_down_when_no_version(make_app):
    fake = FakeOllama(version=None)
    with TestClient(make_app(fake)) as client:
        body = client.get("/healthz").json()
    assert body["status"] == "down"


def test_healthz_degraded_on_partial_gpu(make_app):
    fake = FakeOllama(ps=[{"name": TEST_SETTINGS.model, "size": 100, "size_vram": 40}])
    with TestClient(make_app(fake)) as client:
        body = client.get("/healthz").json()
    assert body["status"] == "degraded"
    assert body["gpu_share"] == 0.4


def test_healthz_degraded_on_model_drift(make_app, fake):
    settings = Settings(api_tokens=("secret",), pinned_digest="sha256:другой")
    with TestClient(make_app(fake, settings)) as client:
        body = client.get("/healthz").json()
    assert body["status"] == "degraded"
    assert "digest" in body["reasons"][0]
    assert fake.calls["model_digest"] == 1  # проверка была на старте (lifespan)


# ---------------------------------------------------------------- metrics
def test_metrics_endpoint_exposes_counters(make_app, fake):
    with TestClient(make_app(fake)) as client:
        client.post("/generate", json={"prompt": "hi"}, headers=AUTH)
        text = client.get("/metrics").text
    assert 'proxy_requests_total{status="ok"} 1.0' in text
    assert "proxy_gen_tokens_per_second_p50 20.0" in text
