"""Переранжирование кандидатов cross-encoder-моделью.

Bi-encoder (эмбеддинги) кодирует запрос и документ НЕЗАВИСИМО — быстро,
но грубо. Cross-encoder читает пару «запрос + документ» ОДНИМ проходом
и оценивает соответствие точнее, но стоит дорого: O(кандидатов) прогонов
модели. Поэтому классическая схема: дешёвый отбор fetch_k кандидатов
эмбеддингами -> точная пересортировка cross-encoder'ом -> top_k.

Аналогия из БД: индекс быстро отдаёт кандидатов, а точная проверка
условия WHERE выполняется только по этим строкам, а не по всей таблице.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from app.services.vectorstore import ScoredChunk

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class RerankerService:
    """Обёртка над CrossEncoder с ленивой загрузкой модели."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model: CrossEncoder | None = None

    def load(self) -> None:
        """Загружает модель (при первом запуске скачивает ~500 МБ)."""
        if self._model is not None:
            return
        from sentence_transformers import CrossEncoder  # ленивый импорт

        logger.info("Загружаю reranker %s ...", self._model_name)
        self._model = CrossEncoder(self._model_name, device="cpu")
        logger.info("Reranker готов")

    def rerank(
        self,
        query: str,
        candidates: list[ScoredChunk],
        top_k: int,
    ) -> list[tuple[ScoredChunk, float]]:
        """Пересортировка кандидатов по оценке cross-encoder.

        Returns:
            Пары (чанк, rerank_score), отсортированные по убыванию оценки.
            Шкала rerank_score НЕ совпадает со шкалой косинусной близости.
        """
        if self._model is None:
            raise RuntimeError("Reranker не загружен: вызовите load()")
        if not candidates:
            return []
        pairs = [(query, candidate.text) for candidate in candidates]
        scores = self._model.predict(pairs)
        ranked = sorted(
            zip(candidates, (float(score) for score in scores), strict=True),
            key=lambda item: item[1],
            reverse=True,
        )
        return ranked[:top_k]

    async def rerank_async(
        self,
        query: str,
        candidates: list[ScoredChunk],
        top_k: int,
    ) -> list[tuple[ScoredChunk, float]]:
        """CPU-bound predict уводим в thread pool, чтобы не блокировать event loop."""
        return await asyncio.to_thread(self.rerank, query, candidates, top_k)
