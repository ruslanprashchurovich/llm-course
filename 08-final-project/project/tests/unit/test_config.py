"""Unit-тесты конфигурации: значения по умолчанию, окружение, валидация."""

from __future__ import annotations

import os

import pytest
from pydantic import ValidationError

from app.config import Settings


@pytest.fixture()
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Убирает все APP_* из окружения, чтобы тесты были воспроизводимыми."""
    for key in list(os.environ):
        if key.startswith("APP_"):
            monkeypatch.delenv(key)


def test_defaults(clean_env: None) -> None:
    settings = Settings(_env_file=None)
    assert settings.environment == "dev"
    assert settings.qdrant_collection == "internal_docs"
    assert settings.search_top_k == 5
    assert settings.search_fetch_k == 20
    assert settings.embedding_dim == 384
    assert settings.llm_think is False
    assert settings.llm_num_ctx == 8192


def test_env_override(clean_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_LLM_MODEL", "llama3.2:1b")
    monkeypatch.setenv("APP_SEARCH_TOP_K", "7")
    monkeypatch.setenv("APP_RERANKER_ENABLED", "false")
    monkeypatch.setenv("APP_LLM_THINK", "true")
    monkeypatch.setenv("APP_LLM_NUM_CTX", "16384")
    settings = Settings(_env_file=None)
    assert settings.llm_model == "llama3.2:1b"
    assert settings.search_top_k == 7
    assert settings.reranker_enabled is False
    assert settings.llm_think is True
    assert settings.llm_num_ctx == 16384


def test_constructor_beats_env(clean_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENVIRONMENT", "prod")
    settings = Settings(_env_file=None, environment="test")
    assert settings.environment == "test"


def test_invalid_value_rejected(clean_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_SEARCH_TOP_K", "0")  # ge=1
    with pytest.raises(ValidationError):
        Settings(_env_file=None)
