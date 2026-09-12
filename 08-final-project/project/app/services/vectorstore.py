"""Работа с векторной базой Qdrant: коллекция, upsert, поиск."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from qdrant_client import AsyncQdrantClient, models

from app.services.chunking import Chunk

logger = logging.getLogger(__name__)


def point_id_for(source: str, chunk_index: int) -> str:
    """Детерминированный UUID точки по источнику и номеру чанка.

    Благодаря uuid5 повторная индексация того же файла ОБНОВЛЯЕТ точки
    (upsert по тому же id), а не плодит дубликаты — как первичный ключ в БД.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}#{chunk_index}"))


@dataclass(frozen=True)
class ScoredChunk:
    """Чанк, найденный в Qdrant, с оценкой близости."""

    text: str
    source: str
    title: str
    chunk_index: int
    score: float


class VectorStore:
    """Асинхронный клиент Qdrant, ограниченный нуждами проекта."""

    def __init__(
        self,
        url: str,
        collection: str,
        vector_size: int,
        timeout_s: float = 10.0,
    ) -> None:
        self._client = AsyncQdrantClient(url=url, timeout=int(timeout_s))
        self._collection = collection
        self._vector_size = vector_size

    async def ensure_collection(self) -> None:
        """Создаёт коллекцию и payload-индекс, если их ещё нет (идемпотентно)."""
        if await self._client.collection_exists(self._collection):
            return
        await self._client.create_collection(
            collection_name=self._collection,
            vectors_config=models.VectorParams(
                size=self._vector_size,
                distance=models.Distance.COSINE,
            ),
        )
        # Индекс по полю source — для фильтров вида «искать только в deploy-guide.md».
        await self._client.create_payload_index(
            collection_name=self._collection,
            field_name="source",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        logger.info(
            "Создана коллекция %s (dim=%d, cosine)", self._collection, self._vector_size
        )

    async def drop_collection(self) -> None:
        """Удаляет коллекцию (для полной переиндексации с чистого листа)."""
        if await self._client.collection_exists(self._collection):
            await self._client.delete_collection(self._collection)
            logger.info("Коллекция %s удалена", self._collection)

    async def upsert_chunks(
        self, chunks: list[Chunk], vectors: list[list[float]]
    ) -> None:
        """Записывает чанки с векторами; повторный вызов обновляет те же точки."""
        points = [
            models.PointStruct(
                id=point_id_for(chunk.source, chunk.chunk_index),
                vector=vector,
                payload={
                    "text": chunk.text,
                    "source": chunk.source,
                    "title": chunk.title,
                    "chunk_index": chunk.chunk_index,
                },
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        await self._client.upsert(
            collection_name=self._collection, points=points, wait=True
        )

    async def search(
        self,
        vector: list[float],
        limit: int,
        source: str | None = None,
    ) -> list[ScoredChunk]:
        """ANN-поиск ближайших чанков, опционально с фильтром по источнику."""
        query_filter = None
        if source:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="source",
                        match=models.MatchValue(value=source),
                    )
                ]
            )
        response = await self._client.query_points(
            collection_name=self._collection,
            query=vector,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        )
        results: list[ScoredChunk] = []
        for point in response.points:
            payload = point.payload or {}
            results.append(
                ScoredChunk(
                    text=str(payload.get("text", "")),
                    source=str(payload.get("source", "")),
                    title=str(payload.get("title", "")),
                    chunk_index=int(payload.get("chunk_index", 0)),
                    score=float(point.score),
                )
            )
        return results

    async def count(self) -> int:
        """Точное число точек в коллекции."""
        result = await self._client.count(collection_name=self._collection, exact=True)
        return int(result.count)

    async def healthy(self) -> bool:
        """True, если Qdrant отвечает и коллекция существует."""
        try:
            return bool(await self._client.collection_exists(self._collection))
        except Exception:  # noqa: BLE001 - health-check намеренно глотает всё
            return False

    async def close(self) -> None:
        """Закрывает сетевые соединения клиента."""
        await self._client.close()
