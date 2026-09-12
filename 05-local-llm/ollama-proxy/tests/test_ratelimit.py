"""Лимит на токен (практика 5.7, средний уровень): юнит-тесты окна + HTTP-контракт.

Время инжектируется (FakeClock) — ни одного sleep: сброс окна проверяется
переводом часов, как Alert в уроке 5.5.
"""

from app.config import Settings
from app.ratelimit import RateLimiter
from fastapi.testclient import TestClient
from tests.conftest import FakeClock

LIMITED = Settings(api_tokens=("alice", "bob"), rate_limit_per_minute=2)


# ---------------------------------------------------------------- юнит: окно
def test_limit_exceeded_gives_honest_retry_after():
    clock = FakeClock()
    limiter = RateLimiter(3, clock=clock)
    assert [limiter.check("t").remaining for _ in range(3)] == [2, 1, 0]
    denied = limiter.check("t")
    assert denied.allowed is False
    assert denied.retry_after_s == 60          # первый запрос выпадет из окна через 60 с
    clock.advance(10)
    assert limiter.check("t").retry_after_s == 50  # Retry-After считается, а не выдумывается


def test_window_slides_without_any_timer():
    clock = FakeClock()
    limiter = RateLimiter(3, clock=clock)
    for _ in range(3):
        limiter.check("t")
    assert limiter.check("t").allowed is False
    clock.advance(60)                         # старые отметки вытеснены при следующей проверке
    assert limiter.check("t").allowed is True


def test_keys_are_independent():
    limiter = RateLimiter(1, clock=FakeClock())
    assert limiter.check("alice").allowed is True
    assert limiter.check("alice").allowed is False
    assert limiter.check("bob").allowed is True   # чужой лимит не тратится


def test_disabled_limiter_allows_everything():
    limiter = RateLimiter(0)
    assert all(limiter.check("t").allowed for _ in range(100))


# ---------------------------------------------------------------- HTTP-контракт
def test_generate_returns_429_with_retry_after(make_app, fake, clock):
    with TestClient(make_app(fake, LIMITED, clock)) as client:
        first = client.post("/generate", json={"prompt": "hi"}, headers={"X-API-Token": "alice"})
        second = client.post("/generate", json={"prompt": "hi"}, headers={"X-API-Token": "alice"})
        third = client.post("/generate", json={"prompt": "hi"}, headers={"X-API-Token": "alice"})
        metrics = client.get("/metrics").text
    assert (first.status_code, second.status_code, third.status_code) == (200, 200, 429)
    assert first.headers["X-RateLimit-Remaining"] == "1"
    assert second.headers["X-RateLimit-Remaining"] == "0"
    assert third.headers["Retry-After"] == "60"
    assert fake.calls["generate"] == 2          # НЕ-вызов: отбитый запрос до Ollama не дошёл
    assert 'proxy_requests_total{status="rate_limited"} 1.0' in metrics


def test_window_reset_lets_requests_through_again(make_app, fake, clock):
    with TestClient(make_app(fake, LIMITED, clock)) as client:
        for _ in range(2):
            client.post("/generate", json={"prompt": "hi"}, headers={"X-API-Token": "alice"})
        assert client.post("/generate", json={"prompt": "hi"},
                           headers={"X-API-Token": "alice"}).status_code == 429
        clock.advance(60)
        assert client.post("/generate", json={"prompt": "hi"},
                           headers={"X-API-Token": "alice"}).status_code == 200


def test_tokens_have_independent_budgets(make_app, fake, clock):
    with TestClient(make_app(fake, LIMITED, clock)) as client:
        for _ in range(2):
            client.post("/generate", json={"prompt": "hi"}, headers={"X-API-Token": "alice"})
        assert client.post("/generate", json={"prompt": "hi"},
                           headers={"X-API-Token": "alice"}).status_code == 429
        assert client.post("/generate", json={"prompt": "hi"},
                           headers={"X-API-Token": "bob"}).status_code == 200


def test_wrong_token_gets_401_not_429(make_app, fake, clock):
    """Порядок зависимостей: auth раньше лимита — чужой токен ничей бюджет не тратит."""
    with TestClient(make_app(fake, LIMITED, clock)) as client:
        for _ in range(5):
            assert client.post("/generate", json={"prompt": "hi"},
                               headers={"X-API-Token": "mallory"}).status_code == 401
        assert client.post("/generate", json={"prompt": "hi"},
                           headers={"X-API-Token": "alice"}).status_code == 200
