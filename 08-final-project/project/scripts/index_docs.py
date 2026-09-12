"""Индексация документов из data/docs в Qdrant.

Запуск из корня проекта (project/):

    python scripts/index_docs.py             # проиндексировать все .md
    python scripts/index_docs.py --recreate  # пересоздать коллекцию с нуля

Скрипт идемпотентен: ID точек детерминированные (uuid5 от source#chunk_index),
поэтому повторный запуск обновляет существующие точки, а не плодит дубли.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

# Позволяем запускать скрипт напрямую: python scripts/index_docs.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.logging_setup import setup_logging  # noqa: E402
from app.services.chunking import Chunk, split_markdown  # noqa: E402
from app.services.embeddings import EmbeddingService  # noqa: E402
from app.services.vectorstore import VectorStore  # noqa: E402

logger = logging.getLogger("scripts.index_docs")


def load_chunks(docs_dir: Path, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Читает все .md из каталога и режет их на чанки."""
    files = sorted(docs_dir.glob("*.md"))
    if not files:
        raise SystemExit(f"В {docs_dir} нет .md-файлов — нечего индексировать")

    chunks: list[Chunk] = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        file_chunks = split_markdown(
            text,
            source=path.name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        logger.info("%-32s -> %d чанков", path.name, len(file_chunks))
        chunks.extend(file_chunks)
    return chunks


async def run(recreate: bool) -> None:
    """Полный цикл индексации: чтение -> чанки -> эмбеддинги -> upsert."""
    settings = get_settings()
    setup_logging(settings.log_level)

    docs_dir = Path(settings.docs_dir)
    chunks = load_chunks(docs_dir, settings.chunk_size, settings.chunk_overlap)
    logger.info("Всего чанков: %d", len(chunks))

    embeddings = EmbeddingService(
        settings.embedding_model, batch_size=settings.embedding_batch_size
    )
    embeddings.load()

    store = VectorStore(
        url=settings.qdrant_url,
        collection=settings.qdrant_collection,
        vector_size=settings.embedding_dim,
        timeout_s=settings.qdrant_timeout_s,
    )
    try:
        if recreate:
            await store.drop_collection()
        await store.ensure_collection()

        start = time.perf_counter()
        vectors = embeddings.embed_passages([chunk.text for chunk in chunks])
        logger.info("Эмбеддинги готовы за %.1f с", time.perf_counter() - start)

        await store.upsert_chunks(chunks, vectors)
        total = await store.count()
        logger.info(
            "Готово: в коллекции '%s' теперь %d точек",
            settings.qdrant_collection,
            total,
        )
    finally:
        await store.close()


def main() -> None:
    """CLI-обёртка."""
    parser = argparse.ArgumentParser(description="Индексация документации в Qdrant")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="удалить коллекцию и создать заново (полная переиндексация)",
    )
    args = parser.parse_args()
    asyncio.run(run(recreate=args.recreate))


if __name__ == "__main__":
    main()
