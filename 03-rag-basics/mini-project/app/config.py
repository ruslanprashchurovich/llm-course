"""Настройки сервиса — всё из переменных окружения (урок 3.2).

Везде 127.0.0.1, а не localhost: Ollama и uvicorn слушают только IPv4,
а httpx сперва пробует IPv6 — на Windows это ~2с штрафа на соединение.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # mini-project/


def _flag(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).strip().lower() not in ("0", "false", "off", "no")


@dataclass(frozen=True)
class Settings:
    qdrant_url: str
    ollama_url: str
    llm_model: str
    emb_backend: str
    collection: str
    data_dir: Path
    state_path: Path
    log_path: Path
    score_threshold: float
    num_predict: int
    guardrails: bool
    agent_mode: bool
    llm_think: bool
    num_ctx: int


def get_settings() -> Settings:
    backend = os.getenv("EMB_BACKEND", "tfidf")
    # «одна коллекция - одна модель» (урок 3.2) как суффикс имени:
    # tfidf живёт в miniproject_kb, e5 - в miniproject_kb_e5, и запрос
    # никогда не уйдёт в коллекцию, набитую векторами другого эмбеддера
    default_collection = "miniproject_kb" + ("" if backend == "tfidf" else f"_{backend}")
    return Settings(
        qdrant_url=os.getenv("QDRANT_URL", "http://127.0.0.1:6333"),
        ollama_url=os.getenv("OLLAMA_URL", "http://127.0.0.1:11434"),
        llm_model=os.getenv("LLM_MODEL", "qwen2.5:3b"),
        emb_backend=backend,
        collection=os.getenv("RAG_COLLECTION", default_collection),
        # по умолчанию - база знаний модуля (14 документов «Векторики»)
        data_dir=Path(os.getenv("DATA_DIR", str(ROOT.parent / "data"))),
        state_path=Path(os.getenv("STATE_PATH", str(ROOT / "tfidf_state.json"))),
        log_path=Path(os.getenv("LOG_PATH", str(ROOT / "logs" / "rag.jsonl"))),
        score_threshold=float(os.getenv("SCORE_THRESHOLD", "0.10")),
        num_predict=int(os.getenv("NUM_PREDICT", "400")),
        guardrails=_flag("GUARDRAILS"),      # рубежи 2-4: цитаты + self-check (урок 3.4)
        agent_mode=_flag("AGENT_MODE"),      # retry поиска с переформулировкой (урок 3.5)
        # thinking-модели (qwen3.5+) иначе рассуждают перед ответом: RAG-ответу
        # нужен текст, не процесс; false безопасен и для qwen2.5:3b
        llm_think=_flag("LLM_THINK", "0"),
        # контекст на ОДИН запрос (API без состояния): system + чанки + вопрос +
        # ответ; дефолт сервера 4096 для RAG впритык, переполнение режется молча
        num_ctx=int(os.getenv("NUM_CTX", "8192")),
    )
