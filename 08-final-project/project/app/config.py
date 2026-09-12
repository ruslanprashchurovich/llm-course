"""Конфигурация приложения.

Все настройки читаются из переменных окружения с префиксом ``APP_``
и/или из файла ``.env`` в корне проекта. Приоритет источников (от высшего
к низшему): аргументы конструктора -> переменные окружения -> .env ->
значения по умолчанию.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Настройки Docs Assistant.

    Каждое поле можно переопределить переменной окружения:
    например, ``llm_model`` -> ``APP_LLM_MODEL``.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="APP_",
        extra="ignore",
    )

    # --- Общие ---
    app_name: str = "docs-assistant"
    environment: Literal["dev", "prod", "test"] = "dev"
    log_level: str = "INFO"

    # --- Qdrant (векторная БД) ---
    # 127.0.0.1 вместо localhost: на Windows httpx сперва пробует ::1 и теряет
    # ~2 секунды на каждое свежее соединение (замер — урок 3.2 курса).
    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_collection: str = "internal_docs"
    qdrant_timeout_s: float = Field(default=10.0, gt=0)

    # --- Ollama (локальный LLM-сервер) ---
    ollama_base_url: str = "http://127.0.0.1:11434"
    llm_model: str = "qwen3.5:9b"
    llm_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(
        default=1024, gt=0, description="num_predict для Ollama"
    )
    llm_timeout_s: float = Field(default=180.0, gt=0)
    # Thinking-модели (qwen3.5+) по умолчанию рассуждают перед ответом — для RAG
    # это лишние секунды и токены. false отправляется явно; модели без thinking
    # (qwen2.5:3b) на "think": false не жалуются — сервер отвергает только true.
    llm_think: bool = False
    # Контекстное окно на ОДИН запрос (API без состояния): system + чанки + вопрос
    # + ответ. Дефолт сервера 4096 для RAG впритык, а переполнение Ollama режет
    # молча с начала — то есть системный промпт. Смена значения = перезагрузка
    # модели (KV-кеш другого размера, урок 5.6).
    llm_num_ctx: int = Field(default=8192, gt=0, description="num_ctx для Ollama")

    # --- Эмбеддинги ---
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_dim: int = Field(default=384, gt=0)
    embedding_batch_size: int = Field(default=32, gt=0)

    # --- Reranker (переранжирование кандидатов) ---
    reranker_enabled: bool = True
    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

    # --- Поиск ---
    search_top_k: int = Field(
        default=5, ge=1, le=20, description="сколько чанков вернуть"
    )
    search_fetch_k: int = Field(
        default=20, ge=1, le=100, description="кандидатов до reranker"
    )
    search_min_score: float = Field(default=0.0, ge=-1.0, le=1.0)

    # --- История ---
    history_db_path: str = "data/history.db"

    # --- Индексация ---
    docs_dir: str = "data/docs"
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)


@lru_cache
def get_settings() -> Settings:
    """Возвращает единственный (закешированный) экземпляр настроек.

    ``lru_cache`` гарантирует, что окружение и ``.env`` читаются один раз
    за жизнь процесса — аналог синглтона пула соединений с БД.
    """
    return Settings()
