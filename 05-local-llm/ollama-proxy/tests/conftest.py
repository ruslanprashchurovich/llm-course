"""Фейковый клиент Ollama со счётчиками вызовов — тесты без сети и без модели.

Тот же приём, что в mini-project модуля 3: фейк считает вызовы, и тесты
проверяют не только ответы, но и НЕ-вызовы (например, что ReadTimeout
не приводит к повторному обращению к Ollama). Плюс FakeClock: лимиты
и окна проверяются без единого sleep (приём урока 5.5).
"""

from __future__ import annotations

import pytest

from app.config import Settings
from app.main import create_app

# «Диск Ollama» по умолчанию: основная и запасная модели курса
DEFAULT_TAGS = [
    {
        "name": "qwen2.5:3b", "size": 1_929_912_432,
        "digest": "sha256:357c53fb659c" + "0" * 52,
        "modified_at": "2026-08-26T10:00:00Z",
        "details": {"parameter_size": "3.1B", "quantization_level": "Q4_K_M"},
    },
    {
        "name": "qwen2.5:1.5b", "size": 986_060_000,
        "digest": "sha256:" + "1" * 64,
        "modified_at": "2026-08-26T10:05:00Z",
        "details": {"parameter_size": "1.5B", "quantization_level": "Q4_K_M"},
    },
]


class FakeOllama:
    """Подменяет OllamaClient: тот же интерфейс, ноль сети.

    Публичные поля ps_models / eval_duration тесты меняют «на лету» —
    так имитируется съезд модели с GPU или падение скорости между запросами.
    """

    def __init__(self, *, version="0.0-fake", ps=None, tags=None, digest="sha256:fake-digest",
                 error: Exception | None = None, tags_error: Exception | None = None,
                 eval_count: int = 40, eval_duration: int = 2_000_000_000,
                 load_duration: int = 20_000_000):
        self._version = version
        self.ps_models: list[dict] = list(ps) if ps is not None else []
        self.tags_models: list[dict] = tags if tags is not None else DEFAULT_TAGS
        self._digest = digest
        self.error = error
        self.tags_error = tags_error
        self.eval_count = eval_count
        self.eval_duration = eval_duration
        self.load_duration = load_duration
        self.calls = {"generate": 0, "version": 0, "ps": 0, "model_digest": 0, "tags": 0}
        self.last_prompt: str | None = None
        self.last_options: dict | None = None
        self.last_model: str | None = None
        self.models_used: list[str] = []

    async def version(self):
        self.calls["version"] += 1
        return self._version

    async def ps(self):
        self.calls["ps"] += 1
        return self.ps_models

    async def tags(self):
        self.calls["tags"] += 1
        if self.tags_error is not None:
            raise self.tags_error
        return self.tags_models

    async def model_digest(self, model: str):
        self.calls["model_digest"] += 1
        return self._digest

    async def generate(self, prompt: str, options: dict, model: str | None = None) -> dict:
        self.calls["generate"] += 1
        self.last_prompt = prompt
        self.last_options = options
        self.last_model = model
        self.models_used.append(model or "fake-model")
        if self.error is not None:
            raise self.error
        return {
            "model": model or "fake-model",
            "response": f"эхо: {prompt[:30]}",
            "prompt_eval_count": 10,
            "eval_count": self.eval_count,
            "eval_duration": self.eval_duration,
            "load_duration": self.load_duration,
            "done_reason": "stop",
        }

    async def aclose(self):
        pass


class FakeClock:
    """Управляемое время для RateLimiter: тесты двигают его сами, без sleep."""

    def __init__(self, start: float = 1_000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


TEST_SETTINGS = Settings(api_tokens=("secret",), num_predict_cap=64,
                         retries=2, backoff_base_s=0.01, queue_timeout_s=0.5)


@pytest.fixture
def fake():
    return FakeOllama()


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def make_app():
    def _make(fake_client: FakeOllama, settings: Settings = TEST_SETTINGS,
              clock: FakeClock | None = None):
        return create_app(client=fake_client, settings=settings, clock=clock)
    return _make
