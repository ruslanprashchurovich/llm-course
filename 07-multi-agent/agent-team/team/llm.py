"""Клиенты LLM с единым интерфейсом ``chat(system, user, num_predict) -> LLMReply``.

* ``OllamaChat``  — /api/chat напрямую (httpx, явные таймауты, ``think`` полем запроса).
* ``ProxyChat``   — через ollama-proxy урока 5.7 (/generate: токен, лимит, потолок
                    num_predict, метрики, деградация) — «финальный штрих» задания 3.
* ``ScriptedLLM`` — детерминированная подмена для тестов: реплики по ролям, ноль сети.

Оркестраторы (императивный, граф, триаж) не знают, кто за интерфейсом, — поэтому
одни и те же тесты гоняют их без Ollama, а живые прогоны — с любой моделью.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import httpx

from team.config import Settings


@dataclass
class LLMReply:
    content: str
    prompt_tokens: int = 0
    answer_tokens: int = 0
    seconds: float = 0.0
    model: str = ""
    done_reason: str = ""
    degraded_to: str | None = None  # заполняет только ProxyChat (капстоун 5.7)


class ChatLLM(Protocol):
    def chat(self, system: str, user: str, *, num_predict: int) -> LLMReply: ...


class OllamaChat:
    """Прямой вызов /api/chat — тот же запрос, что в уроке 7.2, плюс ``think``."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        think: bool = False,
        temperature: float = 0.0,
        timeout_s: float = 300.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.think = think
        self.temperature = temperature
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(connect=5.0, read=timeout_s, write=10.0, pool=5.0),
        )

    def chat(self, system: str, user: str, *, num_predict: int) -> LLMReply:
        started = time.perf_counter()
        response = self._client.post(
            "/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                # think — поле ЗАПРОСА, не options (урок 5.2, частая ошибка 6)
                "think": self.think,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": num_predict,
                },
            },
        )
        response.raise_for_status()
        data = response.json()
        return LLMReply(
            content=data["message"]["content"],
            prompt_tokens=data.get("prompt_eval_count", 0),
            answer_tokens=data.get("eval_count", 0),
            seconds=time.perf_counter() - started,
            model=data.get("model", self.model),
            done_reason=data.get("done_reason", ""),
        )

    def close(self) -> None:
        self._client.close()


class ProxyChat:
    """Те же реплики, но через ollama-proxy: заголовок X-API-Token, /generate.

    Прокси принимает голый ``prompt`` (/api/generate), поэтому system и user
    склеиваем в один текст. 429 с Retry-After — честно ждём и повторяем
    (лимит на токен из практики 5.7); ``degraded_to`` из ответа прокси
    возвращаем наверх — оркестратор увидит, что ответ пришёл от запасной модели.
    """

    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        temperature: float = 0.0,
        timeout_s: float = 300.0,
        max_429_retries: int = 3,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.temperature = temperature
        self.max_429_retries = max_429_retries
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={"X-API-Token": token},
            timeout=httpx.Timeout(connect=5.0, read=timeout_s, write=10.0, pool=5.0),
        )

    def chat(self, system: str, user: str, *, num_predict: int) -> LLMReply:
        started = time.perf_counter()
        payload = {
            "prompt": f"{system}\n\n{user}",
            "num_predict": num_predict,
            "temperature": self.temperature,
        }
        for attempt in range(self.max_429_retries + 1):
            response = self._client.post("/generate", json=payload)
            if response.status_code == 429 and attempt < self.max_429_retries:
                # Retry-After у прокси честный (скользящее окно) — ждём ровно столько
                wait = min(float(response.headers.get("Retry-After", "1")), 30.0)
                time.sleep(wait)
                continue
            break
        response.raise_for_status()
        data = response.json()
        return LLMReply(
            content=data.get("text", ""),
            prompt_tokens=data.get("prompt_tokens", 0),
            answer_tokens=data.get("completion_tokens", 0),
            seconds=time.perf_counter() - started,
            model=data.get("model", ""),
            degraded_to=data.get("degraded_to"),
        )

    def close(self) -> None:
        self._client.close()


@dataclass
class ScriptedCall:
    role: str
    system: str
    user: str
    num_predict: int


class ScriptedLLM:
    """Подмена для тестов: очередь реплик на каждую роль, роль узнаём по system-промпту.

    Когда реплики роли кончились — повторяем последнюю (циклы в тестах не падают
    на IndexError, а упираются в лимиты оркестратора — их и проверяем).
    """

    def __init__(
        self,
        replies: dict[str, list[str]],
        role_of: dict[str, str],
        tokens_per_call: tuple[int, int] = (100, 50),
    ) -> None:
        self._queues = {role: list(items) for role, items in replies.items()}
        self._role_of = dict(role_of)  # system-промпт -> имя роли
        self._tokens = tokens_per_call
        self.calls: list[ScriptedCall] = []

    def chat(self, system: str, user: str, *, num_predict: int) -> LLMReply:
        role = self._role_of.get(system, system)
        self.calls.append(ScriptedCall(role, system, user, num_predict))
        queue = self._queues.get(role)
        if not queue:
            raise AssertionError(f"ScriptedLLM: нет реплик для роли {role!r}")
        content = queue.pop(0) if len(queue) > 1 else queue[0]
        return LLMReply(
            content=content,
            prompt_tokens=self._tokens[0],
            answer_tokens=self._tokens[1],
            seconds=0.0,
            model="scripted",
        )

    def calls_by_role(self, role: str) -> list[ScriptedCall]:
        return [c for c in self.calls if c.role == role]


def make_llm(settings: Settings) -> OllamaChat | ProxyChat:
    """Фабрика по настройкам: TEAM_BACKEND=ollama|proxy."""
    if settings.backend == "proxy":
        return ProxyChat(
            settings.proxy_url,
            settings.proxy_token,
            temperature=settings.temperature,
            timeout_s=settings.timeout_s,
        )
    if settings.backend != "ollama":
        raise ValueError(
            f"TEAM_BACKEND={settings.backend!r}: ожидается ollama или proxy"
        )
    return OllamaChat(
        settings.ollama_url,
        settings.model,
        think=settings.think,
        temperature=settings.temperature,
        timeout_s=settings.timeout_s,
    )
