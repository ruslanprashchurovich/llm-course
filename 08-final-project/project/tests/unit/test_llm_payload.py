"""Unit-тесты сборки запроса к Ollama: флаг think доходит до сервера.

Сеть не нужна: проверяем словарь, который уйдёт в POST /api/chat.
"""

from __future__ import annotations

from app.services.llm import OllamaClient

MESSAGES = [{"role": "user", "content": "Как деплоить?"}]


def test_payload_disables_thinking_by_default() -> None:
    client = OllamaClient(base_url="http://127.0.0.1:11434", model="qwen3.5:9b")
    payload = client._build_payload(MESSAGES, temperature=0.2, max_tokens=64, stream=False)
    # Без явного false thinking-модель рассуждает по умолчанию (nil -> true в Ollama).
    assert payload["think"] is False
    assert payload["model"] == "qwen3.5:9b"
    assert payload["stream"] is False
    # num_ctx не задан -> в options его нет, действует дефолт сервера.
    assert payload["options"] == {"temperature": 0.2, "num_predict": 64}


def test_payload_can_enable_thinking() -> None:
    client = OllamaClient(
        base_url="http://127.0.0.1:11434", model="qwen3.5:9b", think=True, num_ctx=8192
    )
    payload = client._build_payload(MESSAGES, temperature=0.2, max_tokens=64, stream=True)
    assert payload["think"] is True
    assert payload["stream"] is True
    # Окно контекста уходит в options рядом с temperature/num_predict.
    assert payload["options"]["num_ctx"] == 8192
