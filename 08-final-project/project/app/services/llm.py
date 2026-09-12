"""HTTP-клиент для Ollama — локального сервера LLM.

Ollama поднимает REST API на 11434 порту. Мы используем два режима
эндпоинта /api/chat:

* обычный запрос-ответ (stream=false) — один JSON;
* стриминг (stream=true) — NDJSON: по одной JSON-строке на фрагмент ответа.

Поле ``think`` уходит в каждом запросе: thinking-модели (qwen3.5+) иначе
рассуждают перед ответом по умолчанию, а RAG-ответу нужен текст, не процесс.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Ошибка обращения к LLM-серверу (сеть, таймаут, ответ с ошибкой)."""


class OllamaClient:
    """Асинхронный клиент Ollama поверх httpx."""

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_s: float = 180.0,
        think: bool = False,
        num_ctx: int | None = None,
    ) -> None:
        # По умолчанию httpx ждёт всего 5 секунд — для LLM на CPU этого
        # катастрофически мало, поэтому таймаут чтения задаём явно.
        self._model = model
        self._think = think
        self._num_ctx = num_ctx  # None -> не передаём, действует дефолт сервера
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout_s, connect=5.0),
        )

    @property
    def model(self) -> str:
        """Имя модели, с которой работает клиент."""
        return self._model

    def _build_payload(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        stream: bool,
    ) -> dict[str, Any]:
        options: dict[str, Any] = {"temperature": temperature, "num_predict": max_tokens}
        if self._num_ctx is not None:
            # Окно на ОДИН запрос: API без состояния, историю и чанки клиент
            # присылает целиком. Переполнение сервер режет молча с начала.
            options["num_ctx"] = self._num_ctx
        return {
            "model": self._model,
            "messages": messages,
            "stream": stream,
            # Явный false: без поля thinking-модель думает по умолчанию (routes.go
            # Ollama: nil -> true), а модели без thinking false принимают молча.
            "think": self._think,
            "options": options,
        }

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        """Одноразовый запрос: возвращает полный текст ответа модели."""
        payload = self._build_payload(
            messages, temperature=temperature, max_tokens=max_tokens, stream=False
        )
        try:
            response = await self._client.post("/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise LLMError(f"Запрос к Ollama не удался: {exc}") from exc
        if data.get("error"):
            raise LLMError(str(data["error"]))
        return str(data.get("message", {}).get("content", ""))

    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> AsyncIterator[str]:
        """Стриминг: отдаёт фрагменты ответа по мере генерации."""
        payload = self._build_payload(
            messages, temperature=temperature, max_tokens=max_tokens, stream=True
        )
        try:
            async with self._client.stream("POST", "/api/chat", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    if data.get("error"):
                        raise LLMError(str(data["error"]))
                    token = data.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if data.get("done"):
                        break
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            raise LLMError(f"Стриминг из Ollama прервался: {exc}") from exc

    async def healthy(self) -> bool:
        """True, если сервер Ollama отвечает."""
        try:
            response = await self._client.get("/api/tags")
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    async def close(self) -> None:
        """Закрывает пул соединений httpx."""
        await self._client.aclose()
