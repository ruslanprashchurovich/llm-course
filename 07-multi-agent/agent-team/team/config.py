"""Настройки из переменных окружения (правило курса с урока 1.4)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}
_BACKENDS = {"ollama", "proxy"}


class SettingsError(ValueError):
    """Некорректное значение переменной окружения."""


def _str(name: str, default: str) -> str:
    """Строка из env с обрезкой пробелов по краям."""
    raw = os.getenv(name)
    return default if raw is None else raw.strip()


def _flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise SettingsError(f"{name}={raw!r}: ожидалось одно из {sorted(_TRUE | _FALSE)}")


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw.strip())
    except ValueError:
        raise SettingsError(f"{name}={raw!r}: ожидалось число") from None


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except ValueError:
        raise SettingsError(f"{name}={raw!r}: ожидалось целое число") from None


@dataclass(frozen=True)
class Settings:
    # 127.0.0.1, а не localhost: на Windows это минус ~1-2 c на соединение (урок 5.3)
    ollama_url: str = "http://127.0.0.1:11434"
    # Модель модуля 7 — qwen3.5:9b; её провалы — материал уроков. Другая модель —
    # через TEAM_MODEL (у thinking-моделей вроде qwen3.5 оставьте TEAM_THINK=0,
    # иначе рассуждения съедят num_predict — ловушка урока 5.2).
    model: str = "qwen3.5:9b"
    think: bool = False
    temperature: float = 0.0
    timeout_s: float = 300.0

    # Куда ходить за токенами: напрямую в Ollama или через ollama-proxy урока 5.7
    backend: str = "ollama"  # ollama | proxy
    proxy_url: str = "http://127.0.0.1:8005"
    # repr=False: токен не должен попадать в логи и трейсбеки
    proxy_token: str = field(default="dev-token", repr=False, metadata={"secret": True})

    # Лимиты оркестратора — в коде, как всегда (урок 7.2)
    max_fix_rounds: int = 2
    judge_timeout_s: float = 10.0
    # Потолок шагов графа LangGraph (страховка от вечного цикла)
    recursion_limit: int = 50

    def __post_init__(self) -> None:
        # Нормализация. Класс frozen, поэтому присваиваем через object.__setattr__ —
        # это штатная идиома для frozen-датаклассов.
        object.__setattr__(self, "ollama_url", self.ollama_url.rstrip("/"))
        object.__setattr__(self, "proxy_url", self.proxy_url.rstrip("/"))
        object.__setattr__(self, "backend", self.backend.strip().lower())

        # Валидация: лучше упасть здесь с понятным текстом, чем в глубине оркестратора.
        if self.backend not in _BACKENDS:
            raise SettingsError(
                f"backend={self.backend!r}: ожидалось одно из {sorted(_BACKENDS)}"
            )
        if not 0.0 <= self.temperature <= 2.0:
            raise SettingsError(f"temperature={self.temperature}: ожидалось 0.0..2.0")
        if self.timeout_s <= 0 or self.judge_timeout_s <= 0:
            raise SettingsError("таймауты должны быть положительными")
        if self.max_fix_rounds < 0:
            raise SettingsError(f"max_fix_rounds={self.max_fix_rounds}: ожидалось >= 0")
        if self.recursion_limit < 1:
            raise SettingsError(
                f"recursion_limit={self.recursion_limit}: ожидалось >= 1"
            )

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            ollama_url=_str("OLLAMA_URL", cls.ollama_url),
            model=_str("TEAM_MODEL", cls.model),
            think=_flag("TEAM_THINK", cls.think),
            temperature=_float("TEAM_TEMPERATURE", cls.temperature),
            timeout_s=_float("TEAM_TIMEOUT_S", cls.timeout_s),
            backend=_str("TEAM_BACKEND", cls.backend),
            proxy_url=_str("PROXY_URL", cls.proxy_url),
            proxy_token=_str("PROXY_TOKEN", cls.proxy_token),
            max_fix_rounds=_int("TEAM_MAX_FIX_ROUNDS", cls.max_fix_rounds),
            judge_timeout_s=_float("TEAM_JUDGE_TIMEOUT_S", cls.judge_timeout_s),
            recursion_limit=_int("TEAM_RECURSION_LIMIT", cls.recursion_limit),
        )


def _mask(value: str) -> str:
    """Показываем длину и хвост, но не сам секрет."""
    if not value:
        return "<пусто>"
    return f"{'*' * max(len(value) - 4, 0)}{value[-4:]}"


def print_settings(settings: Settings) -> None:
    """Печатает настройки, маскируя поля, помеченные как secret."""
    lines = ["Settings:"]
    for f in fields(settings):
        value = getattr(settings, f.name)
        if f.metadata.get("secret"):
            value = _mask(str(value))
        lines.append(f"  {f.name}: {value}")
    print("\n".join(lines))


if __name__ == "__main__":
    print_settings(Settings.from_env())
