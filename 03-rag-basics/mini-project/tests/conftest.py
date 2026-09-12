"""Фейковые зависимости: тесты работают без Docker, Qdrant и Ollama.

Идея (урок 3.8): RagService и FastAPI-приложение принимают зависимости
снаружи (Deps), поэтому в тестах вместо Qdrant и Ollama - функции-фейки.
Фейки считают вызовы: тесты проверяют не только ответы, но и НЕ-вызовы:
LLM не трогается при отбитом вводе и промахе порога, retry не случается
при чистом self-check.

Фейковый generate различает РОЛЬ вызова по системному промпту: генерация
ответа, self-check (VERIFY_SYSTEM) и переформулировка (REFORMULATE_SYSTEM)
считаются отдельно - без этого тесты на НЕ-вызовы рубежей невозможны.

Логи: autouse-фикстура настраивает structlog на io.StringIO - каждый тест
получает свой буфер, а test_logging.py читает события как данные.
"""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from app.logs import configure_logging
from app.main import Deps, create_app
from app.rag import REFORMULATE_SYSTEM, VERIFY_SYSTEM

FAKE_CHUNKS = [
    {"chunk_id": "otpusk_0", "doc_id": "otpusk",
     "title": "Отпуск в компании",
     "text": "Каждому сотруднику положено 28 календарных дней отпуска в год."},
    {"chunk_id": "otpusk_1", "doc_id": "otpusk",
     "title": "Отпуск в компании",
     "text": "Заявку на отпуск нужно подать минимум за 14 дней до начала."},
    {"chunk_id": "deploy_0", "doc_id": "deploy",
     "title": "Деплой и стенды",
     "text": "Деплой в production разрешён с понедельника по четверг."},
]

# честный ответ: валидная ссылка [1] и дословная цитата из FAKE_CHUNKS[0]
GOOD_ANSWER = ("Положено 28 календарных дней отпуска в год. "
               "<quote>28 календарных дней отпуска в год</quote> [1]")


def make_deps(top_score: float = 0.42,
              answer: str | list[str] = GOOD_ANSWER,
              verify_answer: str = "да",
              reformulated: str = "ежегодный оплачиваемый отпуск сотрудника",
              search_scores: list[float] | None = None,
              healthy: bool = True) -> tuple[Deps, dict]:
    """Фейковые Deps + счётчики вызовов по ролям.

    search_scores - top-скоры по порядку вызовов search (последний
    повторяется), answer - строка или список строк по попыткам генерации.
    """
    calls = {"search": 0, "generate": 0, "self_check": 0, "reformulate": 0,
             "queries": []}
    answers = answer if isinstance(answer, list) else [answer]
    scores = search_scores or [top_score]

    async def search(question: str, top_k: int) -> list[dict]:
        score = scores[min(calls["search"], len(scores) - 1)]
        calls["search"] += 1
        calls["queries"].append(question)
        scored = [dict(chunk, score=round(score - 0.05 * i, 3))
                  for i, chunk in enumerate(FAKE_CHUNKS)]
        return scored[:top_k]

    async def generate(messages: list[dict], **options) -> str:
        system = messages[0]["content"]
        if system == VERIFY_SYSTEM:
            calls["self_check"] += 1
            return verify_answer
        if system == REFORMULATE_SYSTEM:
            calls["reformulate"] += 1
            return reformulated
        text = answers[min(calls["generate"], len(answers) - 1)]
        calls["generate"] += 1
        return text

    async def health() -> dict:
        return {"status": "ok" if healthy else "degraded",
                "points": len(FAKE_CHUNKS), "model": "fake"}

    return Deps(search=search, generate=generate, health=health), calls


@pytest.fixture(autouse=True)
def log_stream():
    """Каждому тесту - свой буфер логов: JSONL читается прямо из него."""
    stream = io.StringIO()
    configure_logging(stream)
    return stream


@pytest.fixture()
def client_and_calls(log_stream):
    """TestClient поверх приложения с фейками (with - чтобы отработал lifespan)."""
    deps, calls = make_deps()
    with TestClient(create_app(deps=deps)) as client:
        yield client, calls


@pytest.fixture()
def client(client_and_calls):
    return client_and_calls[0]
