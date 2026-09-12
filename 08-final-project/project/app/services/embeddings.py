"""Сервис эмбеддингов на базе sentence-transformers (семейство моделей E5).

Важно: импорт ``sentence_transformers`` (и стоящего за ним PyTorch) занимает
секунды и тянет сотни мегабайт — поэтому импорт ленивый, внутри ``load()``.
Тесты и лёгкие скрипты могут импортировать этот модуль мгновенно.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # только для подсказок типов, в рантайме не импортируется
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Обёртка над SentenceTransformer.

    Модели семейства E5 обучены с префиксами: ``query:`` для запросов и
    ``passage:`` для документов. Без префиксов качество поиска заметно падает —
    это зашито в обучение модели, а не наша прихоть.
    """

    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._model: SentenceTransformer | None = None

    def load(self) -> None:
        """Загружает модель в память (при первом запуске скачивает ~120 МБ)."""
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer  # ленивый импорт

        logger.info("Загружаю модель эмбеддингов %s ...", self._model_name)
        self._model = SentenceTransformer(self._model_name, device="cpu")
        logger.info("Модель эмбеддингов готова (dim=%d)", self.dimension)

    @property
    def model(self) -> SentenceTransformer:
        """Загруженная модель; бросает ошибку, если ``load()`` не вызывали."""
        if self._model is None:
            raise RuntimeError("Модель эмбеддингов не загружена: вызовите load()")
        return self._model

    @property
    def dimension(self) -> int:
        """Размерность векторов модели."""
        return int(self.model.get_embedding_dimension() or 0)

    def embed_query(self, text: str) -> list[float]:
        """Эмбеддинг поискового запроса (с префиксом ``query:``)."""
        vector = self.model.encode(f"query: {text}", normalize_embeddings=True)
        return vector.tolist()

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        """Эмбеддинги документов (с префиксом ``passage:``), батчами."""
        vectors = self.model.encode(
            [f"passage: {text}" for text in texts],
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [vector.tolist() for vector in vectors]

    async def embed_query_async(self, text: str) -> list[float]:
        """Асинхронная обёртка.

        ``encode`` — CPU-bound и блокирует поток; в async-приложении уводим
        его в thread pool, иначе на время расчёта встанет весь event loop
        (аналогия: синхронный драйвер БД внутри async-хендлера).
        """
        return await asyncio.to_thread(self.embed_query, text)
