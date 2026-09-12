"""Индексация базы знаний «Векторики» в Qdrant.

Отдельный батч-скрипт, а НЕ код при старте сервиса: индекс живёт дольше
процесса, переиндексация — редкое событие, и запускать её должен человек
(или планировщик), а не каждый рестарт приложения.

Синхронный клиент — сознательно: батч-скрипту некого обслуживать,
пока он ждёт Qdrant, так что асинхронность ему ничего не даёт.

Запуск из папки модуля:
    python service/index_kb.py                 # TF-IDF (работает без скачиваний)
    python service/index_kb.py --backend e5    # когда e5 в кэше HF

Переменные окружения: QDRANT_URL (http://localhost:6333).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # папка модуля 03-rag-basics
sys.path.insert(0, str(ROOT))

from utils import chunk_paragraphs, load_documents          # noqa: E402
from service.embedder import TfidfEmbedder, make_embedder   # noqa: E402

QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")


def build_chunks() -> list[dict]:
    """Документы -> чанки с контекстными заголовками (уроки 2.5-2.6, как в 3.1)."""
    chunks: list[dict] = []
    for doc in load_documents(ROOT / "data"):
        for i, piece in enumerate(chunk_paragraphs(doc.text, max_chars=500)):
            chunks.append({
                "chunk_id": f"{doc.doc_id}_{i}",
                "doc_id": doc.doc_id,
                "title": doc.title,
                "topic": doc.topic,
                "text": piece,
                "indexed_text": piece if i == 0 else f"{doc.title}. {piece}",
            })
    return chunks


def main() -> int:
    parser = argparse.ArgumentParser(description="Индексация БЗ в Qdrant")
    parser.add_argument("--backend", default=os.getenv("EMB_BACKEND", "tfidf"),
                        choices=["tfidf", "e5"])
    args = parser.parse_args()

    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    collection = f"vectorika_rag_{args.backend}"   # одна коллекция - одна модель
    chunks = build_chunks()
    print(f"[1/4] чанки: {len(chunks)} из {ROOT / 'data'}")

    embedder = make_embedder(args.backend)
    embedder.fit([c["indexed_text"] for c in chunks])
    if isinstance(embedder, TfidfEmbedder):
        state_path = Path(__file__).parent / "tfidf_state.json"
        embedder.save(state_path)
        print(f"[2/4] эмбеддер {embedder.name}: dim={embedder.dim}, "
              f"состояние -> {state_path.name}")
    else:
        print(f"[2/4] эмбеддер {embedder.name}: dim={embedder.dim} (без состояния)")

    client = QdrantClient(url=QDRANT_URL)
    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(
        collection,
        vectors_config=VectorParams(size=embedder.dim, distance=Distance.COSINE),
    )
    print(f"[3/4] коллекция {collection!r} создана (cosine)")

    t0 = time.perf_counter()
    vectors = embedder.embed_docs([c["indexed_text"] for c in chunks])
    points = [
        PointStruct(
            # Qdrant принимает только int/UUID id (урок 2.4) - слаг детерминированно
            # превращаем в UUID, чтобы повторный запуск обновлял, а не дублировал
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk["chunk_id"])),
            vector=vec,
            payload={k: chunk[k] for k in
                     ("chunk_id", "doc_id", "title", "topic", "text")},
        )
        for chunk, vec in zip(chunks, vectors)
    ]
    client.upsert(collection, points=points, wait=True)
    total = client.count(collection).count
    print(f"[4/4] загружено точек: {total} за {time.perf_counter() - t0:.2f}с")
    return 0


if __name__ == "__main__":
    sys.exit(main())
