"""Хранилище чанков: обёртка над Qdrant с фильтрами по метаданным.

Почему обёртка, а не «вызовем qdrant-client напрямую»: у любого векторного
хранилища свой синтаксис фильтров и свои единицы близости. Обёртка держит эту
разницу в одном файле — как репозиторий (repository pattern) поверх ORM.

Единицы измерения: коллекция создаётся с метрикой cosine, и Qdrant отдаёт
score = cosine similarity (больше = ближе). Это противоположно Chroma из
урока 2.3, которая отдаёт distance = 1 - cosine (меньше = ближе), — помните
об этом, когда переносите пороги между хранилищами.

Грабли Qdrant из урока 2.4, учтённые здесь:
  * id точки — только int или UUID: человекочитаемый ключ чанка превращаем
    в UUID детерминированно (uuid5), а сам ключ храним в payload;
  * поля, по которым фильтруем, должны иметь payload-индексы — иначе на
    большой коллекции фильтр превращается в полный перебор.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .config import settings

# Пространство имён для uuid5: одинаковый ключ чанка -> одинаковый id точки,
# поэтому переиндексация перезаписывает точки, а не плодит дубликаты.
_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def point_id(chunk_key: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, chunk_key))


@dataclass
class Chunk:
    """Готовый к индексации кусок документа."""

    id: str  # человекочитаемый ключ: tenant:doc_id:vN:iii
    text: str
    metadata: dict[str, Any]


@dataclass
class Hit:
    """Найденный чанк с оценками разных этапов ранжирования."""

    id: str  # человекочитаемый ключ чанка (из payload)
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    similarity: float | None = None  # косинусная близость (плотный поиск)
    bm25_score: float | None = None  # лексический скор
    rrf_score: float | None = None  # после слияния списков
    rerank_score: float | None = None  # логит cross-encoder
    ranks: dict[str, int] = field(default_factory=dict)

    @property
    def doc_id(self) -> str:
        return str(self.metadata.get("doc_id", "?"))

    @property
    def title(self) -> str:
        return str(self.metadata.get("title", self.doc_id))

    @property
    def final_score(self) -> float:
        for value in (
            self.rerank_score,
            self.rrf_score,
            self.similarity,
            self.bm25_score,
        ):
            if value is not None:
                return float(value)
        return 0.0

    def short(self, width: int = 90) -> str:
        text = self.text.replace("\n", " ")
        return text[:width] + ("…" if len(text) > width else "")


def _hit_from_payload(
    chunk_key: str, payload: dict[str, Any], score: float | None
) -> Hit:
    metadata = {k: v for k, v in payload.items() if k not in {"text", "chunk_key"}}
    return Hit(
        id=chunk_key,
        text=str(payload.get("text", "")),
        metadata=metadata,
        similarity=round(float(score), 4) if score is not None else None,
    )


class KnowledgeBase:
    """Обёртка над Qdrant-коллекцией (сервер из docker-compose модуля)."""

    # поля payload, по которым строим фильтры -> им нужны индексы
    INDEXED_FIELDS: dict[str, str] = {
        "tenant_id": "keyword",
        "doc_id": "keyword",
        "department": "keyword",
        "source_type": "keyword",
        "sensitivity": "integer",
        "risk_level": "integer",
        "version": "integer",
        "is_current": "bool",
    }

    def __init__(
        self, url: str | None = None, collection_name: str | None = None
    ) -> None:
        from qdrant_client import QdrantClient

        self.url = url or settings.qdrant_url
        self.collection_name = collection_name or settings.collection
        self.client = QdrantClient(url=self.url, timeout=15)

    # ------------------------------------------------------------------
    def available(self) -> bool:
        """Быстрая проверка живости — годится для /health."""
        try:
            self.client.get_collections()
            return True
        except Exception:  # noqa: BLE001 - сервер недоступен
            return False

    def collection_exists(self) -> bool:
        return bool(self.client.collection_exists(self.collection_name))

    def ensure_collection(self, dim: int, *, recreate: bool = False) -> None:
        """Создаёт коллекцию под заданную размерность (+ payload-индексы).

        recreate=True нужен TF-IDF: после переобучения словаря меняется сама
        размерность векторов, старые точки геометрически несовместимы с новыми.
        """
        from qdrant_client import models

        if recreate and self.collection_exists():
            self.client.delete_collection(self.collection_name)
        if not self.collection_exists():
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=dim, distance=models.Distance.COSINE
                ),
            )
            for field_name, schema in self.INDEXED_FIELDS.items():
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=schema,
                )

    # ------------------------------------------------------------------
    def upsert(
        self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]
    ) -> int:
        """Идемпотентная запись: одинаковый ключ чанка перезаписывает точку.

        Именно поэтому ключ чанка должен быть детерминированным
        (tenant:doc_id:vN:номер), а не uuid4 — иначе переиндексация плодит
        дубликаты, и модель получает один и тот же текст трижды.
        """
        from qdrant_client import models

        if not chunks:
            return 0
        points = [
            models.PointStruct(
                id=point_id(chunk.id),
                vector=list(vector),
                payload={**chunk.metadata, "chunk_key": chunk.id, "text": chunk.text},
            )
            for chunk, vector in zip(chunks, embeddings)
        ]
        self.client.upsert(self.collection_name, points=points, wait=True)
        return len(points)

    def count(self) -> int:
        return int(self.client.count(self.collection_name, exact=True).count)

    def delete_document(self, doc_id: str, version: int | None = None) -> None:
        from qdrant_client import models

        must: list[Any] = [
            models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))
        ]
        if version is not None:
            must.append(
                models.FieldCondition(
                    key="version", match=models.MatchValue(value=version)
                )
            )
        self.client.delete(
            self.collection_name,
            points_selector=models.FilterSelector(filter=models.Filter(must=must)),
            wait=True,
        )

    def reset(self) -> None:
        """Удаляет коллекцию целиком. Для лабораторных экспериментов."""
        if self.collection_exists():
            self.client.delete_collection(self.collection_name)

    # ------------------------------------------------------------------
    def search(
        self,
        query_embedding: Sequence[float],
        k: int = 10,
        flt: Any | None = None,  # qdrant_client.models.Filter
    ) -> list[Hit]:
        """Плотный (векторный) поиск с фильтром по метаданным.

        Фильтр в Qdrant применяется ВО ВРЕМЯ обхода HNSW (filtered search,
        урок 2.4), а не после — поэтому выдача не «дырявится», как при
        post-фильтрации из урока 2.2. Но чем жёстче фильтр, тем меньше
        кандидатов: если отсекается больше ~90% коллекции, берите k с запасом.
        """
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=list(query_embedding),
            limit=k,
            query_filter=flt,
            with_payload=True,
        )
        hits: list[Hit] = []
        for rank, point in enumerate(response.points, start=1):
            payload = dict(point.payload or {})
            hit = _hit_from_payload(
                str(payload.get("chunk_key", point.id)), payload, point.score
            )
            hit.ranks["dense"] = rank
            hits.append(hit)
        return hits

    def fetch_all(self, flt: Any | None = None, limit: int | None = None) -> list[Hit]:
        """Выгружает чанки без векторного поиска — нужно для BM25 и для инспекции."""
        hits: list[Hit] = []
        offset = None
        page_size = 256
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=flt,
                limit=page_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = dict(point.payload or {})
                hits.append(
                    _hit_from_payload(
                        str(payload.get("chunk_key", point.id)), payload, None
                    )
                )
                if limit is not None and len(hits) >= limit:
                    return hits
            if offset is None:
                return hits

    def documents_summary(self) -> list[dict[str, Any]]:
        """Сводка по документам в индексе: версии, отделы, чанки."""
        summary: dict[tuple[str, int], dict[str, Any]] = {}
        for hit in self.fetch_all():
            key = (hit.doc_id, int(hit.metadata.get("version", 1)))
            row = summary.setdefault(
                key,
                {
                    "doc_id": hit.doc_id,
                    "version": int(hit.metadata.get("version", 1)),
                    "title": hit.metadata.get("title"),
                    "department": hit.metadata.get("department"),
                    "sensitivity": hit.metadata.get("sensitivity"),
                    "is_current": hit.metadata.get("is_current"),
                    "source_type": hit.metadata.get("source_type"),
                    "risk_level": hit.metadata.get("risk_level"),
                    "chunks": 0,
                },
            )
            row["chunks"] += 1
        return sorted(summary.values(), key=lambda row: (row["doc_id"], row["version"]))


def unique_hits(groups: Iterable[Sequence[Hit]]) -> dict[str, Hit]:
    """Собирает уникальные чанки из нескольких списков (по ключу чанка)."""
    merged: dict[str, Hit] = {}
    for group in groups:
        for hit in group:
            existing = merged.get(hit.id)
            if existing is None:
                merged[hit.id] = hit
            else:
                if hit.similarity is not None:
                    existing.similarity = max(
                        existing.similarity or 0.0, hit.similarity
                    )
                if hit.bm25_score is not None:
                    existing.bm25_score = max(
                        existing.bm25_score or 0.0, hit.bm25_score
                    )
                existing.ranks.update(hit.ranks)
    return merged


__all__ = ["KnowledgeBase", "Chunk", "Hit", "unique_hits", "point_id"]
