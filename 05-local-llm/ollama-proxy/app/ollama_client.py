"""Асинхронный клиент Ollama с retry по матрице урока 5.3.

Матрица (см. урок 5.3):
  ConnectError / ConnectTimeout  -> retry с backoff+джиттером -> OllamaUnavailable
  HTTP 502/503                   -> retry с backoff+джиттером -> OllamaBusy
  HTTP 4xx/прочие                -> OllamaBadRequest сразу (повтор не чинит конфиг)
  ReadTimeout                    -> НЕ ловим: летит наружу без retry
                                    (повтор удвоил бы нагрузку — «retry storm»)
"""

from __future__ import annotations

import asyncio
import random

import httpx

from .config import Settings


class OllamaError(Exception):
    """Базовая ошибка общения с Ollama."""


class OllamaUnavailable(OllamaError):
    """Демон недоступен (после всех повторов) — наружу это 502."""


class OllamaBusy(OllamaError):
    """Демон перегружен, 502/503 (после всех повторов) — наружу это 503."""


class OllamaBadRequest(OllamaError):
    """4xx/5xx, которые повторять бессмысленно (нет модели, кривой запрос)."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        super().__init__(f"Ollama HTTP {status_code}: {detail}")


class OllamaClient:
    """transport пробрасывается для тестов (httpx.MockTransport) — сеть не нужна."""

    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None):
        self._settings = settings
        self._client = httpx.AsyncClient(
            base_url=settings.ollama_url,
            timeout=httpx.Timeout(connect=settings.connect_timeout_s,
                                  read=settings.read_timeout_s, write=10.0, pool=5.0),
            transport=transport,
        )

    # ------------------------------------------------------------- служебные
    async def version(self) -> str | None:
        """None = демон не отвечает (для healthz)."""
        try:
            resp = await self._client.get("/api/version", timeout=3.0)
            return resp.json().get("version") if resp.status_code == 200 else None
        except httpx.HTTPError:
            return None

    async def ps(self) -> list[dict]:
        try:
            resp = await self._client.get("/api/ps", timeout=5.0)
            return resp.json().get("models", []) if resp.status_code == 200 else []
        except httpx.HTTPError:
            return []

    async def tags(self) -> list[dict]:
        """Модели на диске (/api/tags): имя, размер, digest, details.

        В отличие от version()/ps(), которые «глотают» сеть ради healthz, здесь
        недоступность демона — исключение: для /models это честный 502.
        """
        try:
            resp = await self._client.get("/api/tags", timeout=5.0)
        except httpx.HTTPError as exc:
            raise OllamaUnavailable(str(exc) or exc.__class__.__name__) from exc
        if resp.status_code >= 400:
            raise OllamaBadRequest(resp.status_code, resp.text[:200])
        return resp.json().get("models", [])

    async def model_digest(self, model: str) -> str | None:
        try:
            models = await self.tags()
        except OllamaError:
            return None
        for m in models:
            if m.get("name") == model:
                return m.get("digest")
        return None

    # ------------------------------------------------------------- генерация
    async def generate(self, prompt: str, options: dict, model: str | None = None) -> dict:
        payload = {
            "model": model or self._settings.model,
            "prompt": prompt,
            "stream": False,
            "think": self._settings.think,   # поле запроса, не options (урок 5.2)
            "options": options,
        }
        last_exc: OllamaError | None = None
        for attempt in range(self._settings.retries + 1):
            try:
                resp = await self._client.post("/api/generate", json=payload)
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                last_exc = OllamaUnavailable(str(exc) or exc.__class__.__name__)
            # httpx.ReadTimeout сюда не попадает намеренно — без retry, наружу
            else:
                if resp.status_code in (502, 503):
                    last_exc = OllamaBusy(f"HTTP {resp.status_code}")
                elif resp.status_code >= 400:
                    raise OllamaBadRequest(resp.status_code, resp.text[:200])
                else:
                    return resp.json()
            if attempt < self._settings.retries:
                delay = self._settings.backoff_base_s * (2 ** attempt) * random.uniform(0.5, 1.5)
                await asyncio.sleep(delay)  # джиттер против «thundering herd» (урок 5.3)
        assert last_exc is not None
        raise last_exc

    async def aclose(self) -> None:
        await self._client.aclose()
