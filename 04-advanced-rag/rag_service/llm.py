"""Клиент локальной LLM (Ollama) + оффлайн-заглушка.

Ollama поднимает HTTP-сервер на 127.0.0.1:11434 (не «localhost» — грабля
Windows из урока 3.2). Нам хватает одного эндпоинта POST /api/chat.
Модель qwen2.5:3b занимает ~2 ГБ; на CPU это 8-20 токенов в секунду
(ответ на 150 токенов — 10-20 секунд), на ноутбучном GPU — в разы быстрее.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .config import settings
from .observability import RAG_LLM_TOKENS, log_event


@dataclass
class LLMResult:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    duration_s: float = 0.0
    is_stub: bool = False


class OllamaLLM:
    """Минимальный клиент Ollama на httpx."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout_s: float | None = None,
    ) -> None:
        self.base_url = (base_url or settings.ollama_url).rstrip("/")
        self.model = model or settings.llm_model
        self.timeout_s = timeout_s or settings.llm_timeout_s

    # ------------------------------------------------------------------
    def available(self) -> bool:
        """Быстрая проверка живости — годится для /health."""
        import httpx

        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=2.0)
            return response.status_code == 200
        except Exception:  # noqa: BLE001 - сеть/сервер недоступны
            return False

    def list_models(self) -> list[str]:
        import httpx

        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            response.raise_for_status()
            return [m["name"] for m in response.json().get("models", [])]
        except Exception:  # noqa: BLE001
            return []

    # ------------------------------------------------------------------
    def chat(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        num_ctx: int = 4096,
        json_mode: bool = False,
    ) -> LLMResult:
        """Один синхронный вызов чат-эндпоинта.

        temperature=0 для RAG — осознанный выбор: нам нужна воспроизводимость,
        а не «креативность». Творчество в ответах на вопросы по регламентам
        называется галлюцинацией.
        """
        import httpx

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": num_ctx,
            },
        }
        if json_mode:
            payload["format"] = "json"

        started = time.perf_counter()
        response = httpx.post(
            f"{self.base_url}/api/chat", json=payload, timeout=self.timeout_s
        )
        response.raise_for_status()
        data = response.json()
        duration = time.perf_counter() - started

        prompt_tokens = int(data.get("prompt_eval_count", 0))
        completion_tokens = int(data.get("eval_count", 0))
        RAG_LLM_TOKENS.labels(kind="prompt").inc(prompt_tokens)
        RAG_LLM_TOKENS.labels(kind="completion").inc(completion_tokens)
        log_event(
            "llm.call",
            model=self.model,
            duration_ms=round(duration * 1000, 1),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        return LLMResult(
            text=(data.get("message") or {}).get("content", "").strip(),
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            duration_s=duration,
        )


class StubLLM:
    """Оффлайн-заглушка для случаев, когда Ollama не запущена.

    Это НЕ языковая модель и не симуляция её качества. Заглушка нужна ровно
    для того, чтобы конвейер (поиск -> промпт -> ответ -> проверка ответа)
    можно было прогнать без скачивания весов.

    Поведение управляется параметром obedient:
      * obedient=True  — «наивная» модель: если в контексте есть инструкция,
        заглушка её выполняет (так ведут себя маленькие модели без защиты);
      * obedient=False — модель, следующая системному промпту: отвечает
        цитатой из первого чанка либо честно отказывается.
    """

    def __init__(self, obedient: bool = False, model: str = "stub") -> None:
        self.obedient = obedient
        self.model = model

    def available(self) -> bool:
        return True

    def list_models(self) -> list[str]:
        return [self.model]

    def chat(self, system: str, user: str, **kwargs: Any) -> LLMResult:
        from .security import scan_injection

        started = time.perf_counter()
        report = scan_injection(user)

        if self.obedient and "credential_bait" in report.matched_rules:
            text = (
                "Универсальный пароль для VPN — Vectorika2026!, свежий конфиг "
                "скачайте по ссылке http://vpn-help-vectorika.example.com/config.zip "
                "и запустите install.exe от администратора. "
                "http://exfil.example.com/collect?q=vpn"
            )
        elif "ИСТОЧНИКИ" in user or "[#1]" in user:
            first = user.split("[#1")[-1][:220].strip()
            text = f"По документам: {first} [#1]"
        else:
            text = (
                "В предоставленных документах нет ответа на этот вопрос. "
                "Уточните вопрос или обратитесь в #hr-help."
            )

        duration = time.perf_counter() - started
        return LLMResult(
            text=text,
            model=f"stub({'obedient' if self.obedient else 'safe'})",
            prompt_tokens=len(user) // 4,
            completion_tokens=len(text) // 4,
            duration_s=duration,
            is_stub=True,
        )


def get_llm(prefer_stub: bool = False, obedient_stub: bool = False) -> OllamaLLM | StubLLM:
    """Возвращает Ollama, если она поднята, иначе заглушку."""
    if prefer_stub:
        return StubLLM(obedient=obedient_stub)
    client = OllamaLLM()
    if client.available():
        return client
    log_event("llm.fallback_to_stub", ollama_url=client.base_url, level=30)
    return StubLLM(obedient=obedient_stub)


__all__ = ["OllamaLLM", "StubLLM", "LLMResult", "get_llm"]
