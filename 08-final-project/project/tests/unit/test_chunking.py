"""Unit-тесты чанкинга: чистые функции, никаких моделей и сети."""

from __future__ import annotations

import pytest

from app.services.chunking import extract_title, split_markdown

SAMPLE = (
    "# Гид по деплою\n"
    "\n"
    "Вступление о том, как мы деплоим сервисы.\n"
    "\n"
    "## Стенды\n"
    "\n"
    + ("Подробный абзац про стенды dev, stage и prod. " * 20)
    + "\n\n"
    + ("Ещё один длинный абзац про переменные окружения на стендах. " * 20)
    + "\n"
    "\n"
    "## Откат\n"
    "\n"
    "Откат выполняется джобой rollback в GitLab CI.\n"
)


def test_extract_title() -> None:
    assert extract_title(SAMPLE, fallback="x") == "Гид по деплою"


def test_extract_title_fallback() -> None:
    assert extract_title("просто текст без заголовка", fallback="doc.md") == "doc.md"


def test_chunks_metadata() -> None:
    chunks = split_markdown(SAMPLE, source="deploy.md", chunk_size=400, chunk_overlap=80)
    assert len(chunks) >= 3
    assert all(chunk.source == "deploy.md" for chunk in chunks)
    assert all(chunk.title == "Гид по деплою" for chunk in chunks)
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))


def test_chunks_respect_max_size() -> None:
    for chunk_size in (200, 400, 1000):
        chunks = split_markdown(SAMPLE, source="d.md", chunk_size=chunk_size, chunk_overlap=50)
        assert all(len(chunk.text) <= chunk_size for chunk in chunks), (
            f"есть чанк длиннее {chunk_size}"
        )


def test_short_section_stays_whole() -> None:
    chunks = split_markdown(SAMPLE, source="d.md", chunk_size=1000, chunk_overlap=100)
    rollback_chunks = [chunk for chunk in chunks if "rollback" in chunk.text]
    assert len(rollback_chunks) == 1
    assert rollback_chunks[0].text.startswith("## Откат")


def test_monster_paragraph_hard_split() -> None:
    monster = "# T\n\n" + "x" * 1500
    chunks = split_markdown(monster, source="m.md", chunk_size=300, chunk_overlap=50)
    assert all(len(chunk.text) <= 300 for chunk in chunks)
    # Скользящее окно с шагом 250 должно покрыть все 1500 символов.
    assert sum(len(chunk.text) for chunk in chunks) >= 1500


def test_overlap_must_be_smaller_than_size() -> None:
    with pytest.raises(ValueError):
        split_markdown(SAMPLE, source="d.md", chunk_size=300, chunk_overlap=300)


def test_empty_document() -> None:
    assert split_markdown("", source="empty.md") == []
