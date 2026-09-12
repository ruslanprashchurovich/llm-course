"""Оркестрация RAG-пайплайна: поиск -> промпт -> генерация.

Слой, который связывает эмбеддинги, Qdrant, reranker и LLM в единый
сценарий. Роутеры вызывают только этот сервис.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.services.vectorstore import ScoredChunk, VectorStore

if TYPE_CHECKING:
    from app.config import Settings
    from app.services.embeddings import EmbeddingService
    from app.services.llm import OllamaClient
    from app.services.reranker import RerankerService

logger = logging.getLogger(__name__)

# Канонический текст отказа. Одна константа обслуживает и промпт, и детектор
# отказа в ответе (schemas.GenerateResponse.is_refusal): разъехаться они не могут.
REFUSAL_MARKER = "В документации ответа не нашёл"

SYSTEM_PROMPT = (
    "Ты — ассистент разработчика по внутренней документации компании. "
    "Отвечай на русском языке, кратко и по делу. "
    "Используй ТОЛЬКО факты из блока «Контекст». "
    "После каждого утверждения ставь ссылку на номер источника в квадратных скобках, например [1]. "
    f"Если в контексте нет ответа на вопрос, честно скажи: «{REFUSAL_MARKER}» — "
    "и не придумывай ничего от себя."
)


@dataclass(frozen=True)
class RankedChunk:
    """Чанк после (возможного) переранжирования."""

    chunk: ScoredChunk
    rerank_score: float | None


def build_context(chunks: list[ScoredChunk]) -> str:
    """Собирает блок «Контекст» с пронумерованными источниками.

    Нумерация [1], [2], ... позволяет модели ссылаться на источники,
    а нам — восстановить, какой файл стоит за каждой ссылкой.
    """
    if not chunks:
        return "Контекст пуст: подходящих документов не найдено."
    blocks = []
    for number, chunk in enumerate(chunks, start=1):
        blocks.append(
            f"[{number}] Источник: {chunk.source} — {chunk.title}\n{chunk.text}"
        )
    return "\n\n".join(blocks)


def build_messages(question: str, chunks: list[ScoredChunk]) -> list[dict[str, str]]:
    """Формирует messages для chat-API: system-инструкция + вопрос с контекстом."""
    context = build_context(chunks)
    user_content = f"Контекст:\n{context}\n\nВопрос: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


class RagService:
    """Фасад RAG-пайплайна: единая точка входа для роутеров."""

    def __init__(
        self,
        settings: Settings,
        embeddings: EmbeddingService,
        vectorstore: VectorStore,
        llm: OllamaClient,
        reranker: RerankerService | None = None,
    ) -> None:
        self._settings = settings
        self._embeddings = embeddings
        self._vectorstore = vectorstore
        self._llm = llm
        self._reranker = reranker

    @property
    def model_name(self) -> str:
        """Имя LLM, которой генерируем ответы (для истории и метаданных)."""
        return self._llm.model

    async def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        fetch_k: int | None = None,
        use_reranker: bool | None = None,
        source: str | None = None,
    ) -> list[RankedChunk]:
        """Семантический поиск: эмбеддинг -> ANN в Qdrant -> (опц.) reranker.

        Args:
            query: текст запроса пользователя.
            top_k: сколько результатов вернуть (None -> из настроек).
            fetch_k: сколько кандидатов брать из Qdrant до reranker'а
                (None -> из настроек; без reranker'а не используется).
            use_reranker: принудительно включить/выключить reranker
                (None -> как в настройках).
            source: ограничить поиск одним файлом-источником.
        """
        settings = self._settings
        top_k = top_k or settings.search_top_k
        want_rerank = (
            settings.reranker_enabled if use_reranker is None else use_reranker
        )
        rerank = want_rerank and self._reranker is not None
        # Для reranker'а забираем больше кандидатов, чем отдадим наружу:
        # дешёвый ANN-отбор широкой сетью, дорогая точная оценка — по кандидатам.
        candidates_k = (
            max(fetch_k or settings.search_fetch_k, top_k) if rerank else top_k
        )

        vector = await self._embeddings.embed_query_async(query)
        candidates = await self._vectorstore.search(
            vector, limit=candidates_k, source=source
        )
        candidates = [c for c in candidates if c.score >= settings.search_min_score]

        if rerank and self._reranker is not None and candidates:
            ranked = await self._reranker.rerank_async(query, candidates, top_k)
            return [
                RankedChunk(chunk=chunk, rerank_score=score) for chunk, score in ranked
            ]
        return [
            RankedChunk(chunk=chunk, rerank_score=None) for chunk in candidates[:top_k]
        ]

    async def answer(
        self,
        question: str,
        *,
        top_k: int | None = None,
        temperature: float | None = None,
    ) -> tuple[str, list[ScoredChunk]]:
        """Полный цикл без стриминга: поиск -> промпт -> готовый ответ."""
        ranked = await self.search(question, top_k=top_k)
        chunks = [item.chunk for item in ranked]
        messages = build_messages(question, chunks)
        answer_text = await self._llm.chat(
            messages,
            temperature=temperature
            if temperature is not None
            else self._settings.llm_temperature,
            max_tokens=self._settings.llm_max_tokens,
        )
        return answer_text, chunks

    async def stream_answer(
        self,
        question: str,
        *,
        top_k: int | None = None,
        temperature: float | None = None,
    ) -> tuple[list[ScoredChunk], AsyncIterator[str]]:
        """Стриминговый вариант: возвращает найденные чанки и итератор токенов.

        Чанки возвращаем сразу (ещё до генерации), чтобы роутер мог отправить
        клиенту событие sources первым — пользователь видит источники,
        пока модель только начинает печатать.
        """
        ranked = await self.search(question, top_k=top_k)
        chunks = [item.chunk for item in ranked]
        messages = build_messages(question, chunks)
        iterator = self._llm.stream_chat(
            messages,
            temperature=temperature
            if temperature is not None
            else self._settings.llm_temperature,
            max_tokens=self._settings.llm_max_tokens,
        )
        return chunks, iterator
