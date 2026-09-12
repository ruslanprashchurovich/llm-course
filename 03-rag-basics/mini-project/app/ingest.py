"""Индексация базы знаний в Qdrant: python -m app.ingest

Отдельный батч-скрипт с синхронным клиентом - сознательно (урок 3.2):
индекс живёт дольше процесса, переиндексация - решение, а не побочный
эффект старта сервиса.
"""

from __future__ import annotations

import sys
import time
import uuid

from .config import get_settings
from .embedder import make_embedder
from .kb import build_chunks


def main() -> int:
    settings = get_settings()

    # ленивые импорты: qdrant-client нужен только индексации и проду,
    # тесты проекта живут без него
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    chunks = build_chunks(settings.data_dir)
    if not chunks:
        print(f"[!] в {settings.data_dir} нет .txt-документов")
        return 1
    print(f"[1/4] чанки: {len(chunks)} из {settings.data_dir}")

    # бэкенд выбирает EMB_BACKEND: tfidf учится на корпусе и сохраняет
    # состояние, e5 берёт веса из кэша HF (fit и save у него - заглушки)
    embedder = make_embedder(settings.emb_backend)
    embedder.fit([c["indexed_text"] for c in chunks])
    embedder.save(settings.state_path)
    state_note = (f"состояние -> {settings.state_path.name}"
                  if settings.emb_backend == "tfidf"
                  else "состояние не требуется (веса в кэше HF)")
    print(f"[2/4] эмбеддер: {embedder.name}, dim={embedder.dim}, {state_note}")

    client = QdrantClient(url=settings.qdrant_url)
    if client.collection_exists(settings.collection):
        client.delete_collection(settings.collection)
    client.create_collection(
        settings.collection,
        vectors_config=VectorParams(size=embedder.dim, distance=Distance.COSINE),
    )
    print(f"[3/4] коллекция {settings.collection!r} создана (cosine)")

    t0 = time.perf_counter()
    vectors = embedder.embed_docs([c["indexed_text"] for c in chunks])
    points = [
        PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk["chunk_id"])),
            vector=vector,
            payload={k: chunk[k] for k in
                     ("chunk_id", "doc_id", "title", "text")},
        )
        for chunk, vector in zip(chunks, vectors)
    ]
    client.upsert(settings.collection, points=points, wait=True)
    total = client.count(settings.collection).count
    print(f"[4/4] загружено точек: {total} за {time.perf_counter() - t0:.2f}с")
    return 0


if __name__ == "__main__":
    sys.exit(main())
