"""Настройки: суффикс бэкенда в имени коллекции и флаги рубежей.

Правило «одна коллекция - одна модель» (урок 3.2) живёт в get_settings:
запрос, закодированный не тем эмбеддером, не падает с ошибкой - он молча
возвращает мусор, поэтому имя коллекции обязано зависеть от бэкенда.
"""

from app.config import get_settings


def test_defaults_tfidf_collection_and_enabled_flags(monkeypatch):
    for name in ("EMB_BACKEND", "RAG_COLLECTION", "GUARDRAILS", "AGENT_MODE", "LLM_THINK", "NUM_CTX"):
        monkeypatch.delenv(name, raising=False)
    settings = get_settings()
    assert settings.emb_backend == "tfidf"
    assert settings.collection == "miniproject_kb"
    assert settings.guardrails is True
    assert settings.agent_mode is True
    assert settings.llm_think is False   # рассуждения thinking-моделей выключены
    assert settings.num_ctx == 8192      # окно на один запрос: 4096 сервера впритык


def test_e5_backend_gets_its_own_collection(monkeypatch):
    monkeypatch.setenv("EMB_BACKEND", "e5")
    monkeypatch.delenv("RAG_COLLECTION", raising=False)
    assert get_settings().collection == "miniproject_kb_e5"


def test_explicit_collection_overrides_suffix(monkeypatch):
    monkeypatch.setenv("EMB_BACKEND", "e5")
    monkeypatch.setenv("RAG_COLLECTION", "my_custom_kb")
    assert get_settings().collection == "my_custom_kb"


def test_flags_understand_zero_and_off(monkeypatch):
    monkeypatch.setenv("GUARDRAILS", "0")
    monkeypatch.setenv("AGENT_MODE", "off")
    monkeypatch.setenv("LLM_THINK", "1")
    monkeypatch.setenv("NUM_CTX", "16384")
    settings = get_settings()
    assert settings.guardrails is False
    assert settings.agent_mode is False
    assert settings.llm_think is True
    assert settings.num_ctx == 16384
