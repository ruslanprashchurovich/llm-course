"""Две модели и деградация (капстоун 5.7): роутер отдельно, HTTP-цикл на фейке.

Фейк отдаёт 40 токенов за 2 с = 20 ток/с; долю GPU и скорость тесты меняют
между запросами через публичные поля ps_models / eval_duration.
"""

from app.config import Settings
from app.routing import ModelRouter
from fastapi.testclient import TestClient
from tests.conftest import FakeOllama

AUTH = {"X-API-Token": "secret"}
PRIMARY, FALLBACK = "qwen2.5:3b", "qwen2.5:1.5b"


def make_router(fallback=FALLBACK, baseline=80.0, **kw) -> ModelRouter:
    return ModelRouter(PRIMARY, fallback, baseline_tps=baseline,
                       gpu_share_floor=0.9, p50_ratio=0.5, **kw)


def gen(client, token=AUTH):
    return client.post("/generate", json={"prompt": "hi"}, headers=token).json()


# ---------------------------------------------------------------- юнит: роутер
def test_gpu_spill_switches_to_fallback_and_back():
    router = make_router()
    state = router.evaluate(gpu_share=0.4, p50_tps=80.0, samples=10)
    assert state.degraded_to == FALLBACK and state.active_model == FALLBACK
    assert "VRAM" in state.reason
    state = router.evaluate(gpu_share=1.0, p50_tps=80.0, samples=10)
    assert state.degraded_to is None and state.active_model == PRIMARY and state.reason is None


def test_slow_p50_switches_and_recovers():
    router = make_router()
    assert router.evaluate(gpu_share=1.0, p50_tps=30.0, samples=10).degraded_to == FALLBACK
    assert router.evaluate(gpu_share=1.0, p50_tps=70.0, samples=10).degraded_to is None


def test_p50_verdict_is_sticky_until_enough_fresh_samples():
    router = make_router()
    router.evaluate(gpu_share=1.0, p50_tps=30.0, samples=10)
    # окно сброшено, свежих ответов мало — прежний вердикт держится
    assert router.evaluate(gpu_share=1.0, p50_tps=None, samples=0).degraded_to == FALLBACK
    assert router.evaluate(gpu_share=1.0, p50_tps=90.0, samples=2).degraded_to == FALLBACK
    assert router.evaluate(gpu_share=1.0, p50_tps=90.0, samples=3).degraded_to is None


def test_baseline_learned_only_on_healthy_primary():
    router = make_router(baseline=None)
    router.evaluate(gpu_share=0.4, p50_tps=15.0, samples=5)   # больная — не учим
    assert router.baseline_tps is None
    router.evaluate(gpu_share=1.0, p50_tps=82.0, samples=5)   # здоровая — запомнили
    assert router.baseline_tps == 82.0
    assert router.evaluate(gpu_share=1.0, p50_tps=30.0, samples=5).degraded_to == FALLBACK


def test_without_fallback_only_reports():
    router = make_router(fallback=None)
    state = router.evaluate(gpu_share=0.4, p50_tps=80.0, samples=10)
    assert state.degraded_to is None and state.active_model == PRIMARY
    assert state.reason is not None                       # причина видна, переключать некуда
    assert router.pick().kind == "primary"


def test_every_fifth_degraded_request_is_a_canary():
    router = make_router()
    router.evaluate(gpu_share=1.0, p50_tps=30.0, samples=10)
    kinds = [router.pick().kind for _ in range(10)]
    assert kinds.count("canary") == 2 and kinds[4] == "canary" and kinds[9] == "canary"
    assert kinds.count("fallback") == 8


# ---------------------------------------------------------------- HTTP-цикл
def test_gpu_degradation_roundtrip_via_healthz(make_app):
    fake = FakeOllama(ps=[{"name": PRIMARY, "size": 100, "size_vram": 100}])
    settings = Settings(api_tokens=("secret",), fallback_model=FALLBACK, baseline_tps=20.0)
    with TestClient(make_app(fake, settings)) as client:
        body = gen(client)
        assert (body["model"], body["degraded_to"], body["route"]) == (PRIMARY, None, "primary")

        fake.ps_models[0]["size_vram"] = 40                 # основная съехала с GPU
        health = client.get("/healthz").json()
        assert health["status"] == "degraded"
        assert health["routing"]["degraded_to"] == FALLBACK
        assert "VRAM" in health["reasons"][0] and FALLBACK in health["reasons"][0]
        assert "proxy_degraded 1.0" in client.get("/metrics").text

        body = gen(client)
        assert (body["model"], body["degraded_to"], body["route"]) == (FALLBACK, FALLBACK, "fallback")
        assert fake.last_model == FALLBACK

        fake.ps_models[0]["size_vram"] = 100                # вернулась в VRAM
        health = client.get("/healthz").json()
        assert health["status"] == "ok" and health["routing"]["degraded_to"] is None
        assert "proxy_degraded 0.0" in client.get("/metrics").text

        body = gen(client)
        assert (body["model"], body["degraded_to"]) == (PRIMARY, None)
    assert fake.models_used == [PRIMARY, FALLBACK, PRIMARY]


def test_p50_degradation_and_canary_recovery(make_app):
    fake = FakeOllama(ps=[{"name": PRIMARY, "size": 100, "size_vram": 100}])  # 20 ток/с
    settings = Settings(api_tokens=("secret",), fallback_model=FALLBACK, baseline_tps=80.0)
    with TestClient(make_app(fake, settings)) as client:
        for _ in range(3):                                   # набираем окно основной
            gen(client)
        health = client.get("/healthz").json()
        assert health["status"] == "degraded"
        assert "p50" in health["reasons"][0] and health["routing"]["degraded_to"] == FALLBACK

        fake.eval_duration = 500_000_000                    # основная «выздоровела»: 80 ток/с
        routes = [gen(client)["route"] for _ in range(15)]
        assert routes.count("fallback") == 12 and routes.count("canary") == 3

        health = client.get("/healthz").json()               # 3 канарейки по 80 ток/с — хватит
        assert health["status"] == "ok" and health["routing"]["degraded_to"] is None
        assert gen(client)["route"] == "primary"
    assert fake.models_used.count(FALLBACK) == 12


def test_degraded_status_without_fallback_keeps_primary(make_app):
    fake = FakeOllama(ps=[{"name": PRIMARY, "size": 100, "size_vram": 40}])
    with TestClient(make_app(fake)) as client:               # TEST_SETTINGS: запасной нет
        health = client.get("/healthz").json()
        assert health["status"] == "degraded"
        assert health["routing"]["degraded_to"] is None
        assert "запасная модель не задана" in health["reasons"][0]
        body = gen(client)
        assert (body["model"], body["degraded_to"]) == (PRIMARY, None)
