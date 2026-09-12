"""Конфигурация RAG-сервиса.

Всё настраивается переменными окружения (12-factor style): в ноутбуках удобно
переопределять значения через os.environ до импорта, а в Docker — через env-файл.

Дефолты продолжают сквозную линию курса: Qdrant в Docker (урок 2.4),
TF-IDF-эмбеддер без скачиваний (уроки 2.5, 3.2), qwen2.5:3b в Ollama.
Адреса — 127.0.0.1, не localhost: на Windows httpx сначала пробует IPv6 ::1
и теряет ~2 секунды на каждом свежем соединении (грабля из урока 3.2).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Корень модуля: .../04-advanced-rag
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Настройки пайплайна. Значения по умолчанию рассчитаны на учебный стенд."""

    # --- эмбеддинги ---
    # "tfidf" — свой эмбеддер из уроков 2.5/3.2: работает сразу, без скачиваний.
    # "e5"    — intfloat/multilingual-e5-small (~470 МБ с Hugging Face): включайте,
    #           если модель уже скачана. Правило «одна коллекция — одна модель»
    #           решается суффиксом в имени коллекции (см. collection ниже).
    emb_backend: str = os.getenv("RAG_EMB_BACKEND", "tfidf")
    e5_model: str = os.getenv("RAG_E5_MODEL", "intfloat/multilingual-e5-small")
    # Состояние TF-IDF (словарь + IDF) версионируется вместе с коллекцией:
    # запрос обязан кодироваться тем же словарём, что и документы.
    tfidf_state_path: str = os.getenv(
        "RAG_TFIDF_STATE", str(BASE_DIR / "tfidf_state.json")
    )

    # --- reranker (урок 4.4; ~470 МБ, качается с Hugging Face по явной команде) ---
    reranker_model: str = os.getenv(
        "RAG_RERANKER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    )

    # --- LLM ---
    ollama_url: str = os.getenv("RAG_OLLAMA_URL", "http://127.0.0.1:11434")
    llm_model: str = os.getenv("RAG_LLM_MODEL", "qwen2.5:3b")
    llm_timeout_s: float = _env_float("RAG_LLM_TIMEOUT_S", 120.0)

    # --- хранилище ---
    qdrant_url: str = os.getenv("RAG_QDRANT_URL", "http://127.0.0.1:6333")
    # суффикс бэкенда не даёт смешать вектора разных моделей в одной коллекции
    collection: str = os.getenv(
        "RAG_COLLECTION", f"vectorika_adv_{os.getenv('RAG_EMB_BACKEND', 'tfidf')}"
    )

    # --- чанкинг ---
    chunk_max_chars: int = _env_int("RAG_CHUNK_MAX_CHARS", 700)
    chunk_overlap_chars: int = _env_int("RAG_CHUNK_OVERLAP", 120)

    # --- поиск ---
    candidates_k: int = _env_int("RAG_CANDIDATES_K", 20)  # сколько тянем из индекса
    final_k: int = _env_int("RAG_FINAL_K", 4)  # сколько кладём в промпт
    # Порог отсечения по косинусу. ВАЖНО: порог не переносится между эмбеддерами
    # (урок 3.4): для TF-IDF годные скоры 0.15-0.45, для e5 — 0.75-0.90.
    min_similarity: float = _env_float("RAG_MIN_SIMILARITY", 0.10)
    min_rerank_score: float = _env_float("RAG_MIN_RERANK_SCORE", 0.0)
    enable_rerank: bool = _env_bool("RAG_ENABLE_RERANK", False)  # урок 4.4
    enable_multiquery: bool = _env_bool("RAG_ENABLE_MULTIQUERY", False)  # урок 4.4
    enable_bm25: bool = _env_bool("RAG_ENABLE_BM25", True)

    # --- безопасность ---
    index_untrusted: bool = _env_bool(
        "RAG_INDEX_UNTRUSTED", True
    )  # индексировать ли UGC
    block_high_risk_chunks: bool = _env_bool("RAG_BLOCK_HIGH_RISK", True)
    mask_pii_in_context: bool = _env_bool("RAG_MASK_PII_CONTEXT", True)
    mask_pii_in_logs: bool = _env_bool("RAG_MASK_PII_LOGS", True)

    # --- прочее ---
    max_context_chars: int = _env_int("RAG_MAX_CONTEXT_CHARS", 4000)
    log_level: str = os.getenv("RAG_LOG_LEVEL", "INFO")
    service_name: str = os.getenv("RAG_SERVICE_NAME", "vectorika-rag")


settings = Settings()

__all__ = ["settings", "Settings", "BASE_DIR", "DATA_DIR"]
