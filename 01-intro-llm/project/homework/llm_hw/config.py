"""Конфигурация из окружения и общий клиент — как в базовом проекте (урок 1.4).

Смена провайдера (Ollama -> облако) остаётся правкой окружения, а не кода:
достаточно указать LLM_BASE_URL, LLM_MODEL и LLM_API_KEY облачного API.
"""

from __future__ import annotations

import os

from openai import OpenAI

BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
MODEL = os.getenv("LLM_MODEL", "qwen2.5:3b")
API_KEY = os.getenv("LLM_API_KEY", "ollama")  # локально не проверяется
TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "120"))

client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
    timeout=TIMEOUT_S,
    max_retries=2,  # транзиентное (сеть, 429, 5xx) SDK ретраит сам — урок 1.4
)


def ollama_root() -> str:
    """http://host:11434/v1 -> http://host:11434 — корень родного API Ollama."""
    return BASE_URL.rsplit("/v1", 1)[0]
