"""Конфигурация из переменных окружения (правило курса с урока 1.4)."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # 127.0.0.1, а не localhost: на Windows это минус ~1-2 c на соединение (урок 5.3)
    ollama_url: str = "http://127.0.0.1:11434"
    model: str = "qwen2.5:3b"
    # Если задан — при старте сверяем с /api/tags; расхождение = model_drift (урок 5.6)
    pinned_digest: str = ""

    # Таймауты раздельные: connect короткий, read длинный (урок 5.3)
    connect_timeout_s: float = 5.0
    read_timeout_s: float = 180.0
    retries: int = 2
    backoff_base_s: float = 0.3

    # Конкуренция: на CPU параллелизм не даёт throughput (уроки 5.3, 5.6)
    max_concurrent: int = 1
    queue_timeout_s: float = 30.0

    # Потолок генерации на запрос — как LIMIT в SQL (урок 5.6)
    num_predict_cap: int = 512

    # Thinking-модели (qwen3.5 и т.п.) по умолчанию рассуждают и съедают num_predict
    # (ловушка урока 5.2). think — поле ЗАПРОСА Ollama; False принимают все модели.
    think: bool = False

    # Токены доступа через запятую; пустая строка = аутентификация выключена (для dev)
    api_tokens: tuple[str, ...] = ("dev-token",)

    # Лимит запросов к /generate на один токен за скользящие 60 с; 0 = выключен
    # (практика 5.7, средний уровень; реализация — app/ratelimit.py)
    rate_limit_per_minute: int = 60

    # Капстоун 5.7: запасная модель и пороги деградации (уроки 5.5, 5.6; app/routing.py)
    fallback_model: str = ""              # пусто = не переключаемся, только статус degraded
    baseline_tps: float = 0.0             # p50 ток/с здоровой основной; 0 = выучить по первым ответам
    degrade_gpu_share_below: float = 0.9  # доля модели в VRAM ниже порога = «съехала с GPU»
    degrade_p50_ratio: float = 0.5        # p50 ниже этой доли baseline = деградация

    @classmethod
    def from_env(cls) -> "Settings":
        tokens = tuple(t.strip() for t in os.getenv("PROXY_TOKENS", "dev-token").split(",")
                       if t.strip())
        return cls(
            ollama_url=os.getenv("OLLAMA_URL", cls.ollama_url),
            model=os.getenv("PROXY_MODEL", cls.model),
            pinned_digest=os.getenv("PROXY_PINNED_DIGEST", cls.pinned_digest),
            connect_timeout_s=float(os.getenv("PROXY_CONNECT_TIMEOUT_S", cls.connect_timeout_s)),
            read_timeout_s=float(os.getenv("PROXY_READ_TIMEOUT_S", cls.read_timeout_s)),
            retries=int(os.getenv("PROXY_RETRIES", cls.retries)),
            max_concurrent=int(os.getenv("PROXY_MAX_CONCURRENT", cls.max_concurrent)),
            queue_timeout_s=float(os.getenv("PROXY_QUEUE_TIMEOUT_S", cls.queue_timeout_s)),
            num_predict_cap=int(os.getenv("PROXY_NUM_PREDICT_CAP", cls.num_predict_cap)),
            think=os.getenv("PROXY_THINK", "0").strip().lower() in {"1", "true", "yes", "on"},
            api_tokens=tokens,
            rate_limit_per_minute=int(os.getenv("PROXY_RATE_LIMIT_PER_MIN",
                                                cls.rate_limit_per_minute)),
            fallback_model=os.getenv("PROXY_FALLBACK_MODEL", cls.fallback_model),
            baseline_tps=float(os.getenv("PROXY_BASELINE_TPS", cls.baseline_tps)),
            degrade_gpu_share_below=float(os.getenv("PROXY_DEGRADE_GPU_SHARE",
                                                    cls.degrade_gpu_share_below)),
            degrade_p50_ratio=float(os.getenv("PROXY_DEGRADE_P50_RATIO", cls.degrade_p50_ratio)),
        )
