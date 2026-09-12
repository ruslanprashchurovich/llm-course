"""Разбиение markdown-документов на чанки для индексации.

Стратегия двухуровневая:

1. Документ режется на секции по заголовкам второго уровня (``## ...``) —
   в хорошо структурированной документации это готовые смысловые блоки.
2. Секции длиннее ``chunk_size`` дорезаются по абзацам с перекрытием
   (overlap), чтобы мысль, разрезанная на границе, целиком попала хотя бы
   в один из соседних чанков.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """Фрагмент документа, готовый к индексации."""

    text: str
    source: str
    title: str
    chunk_index: int


def extract_title(markdown: str, fallback: str) -> str:
    """Возвращает первый заголовок первого уровня (``# ...``) или ``fallback``."""
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return fallback


def _split_sections(markdown: str) -> list[str]:
    """Делит документ на секции по заголовкам ``## ``."""
    sections: list[list[str]] = [[]]
    for line in markdown.splitlines():
        if line.startswith("## ") and any(existing.strip() for existing in sections[-1]):
            sections.append([line])
        else:
            sections[-1].append(line)

    result: list[str] = []
    for lines in sections:
        text = "\n".join(lines).strip()
        if text:
            result.append(text)
    return result


def _hard_split(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Жёстко режет сверхдлинный текст скользящим окном (без учёта структуры)."""
    pieces: list[str] = []
    step = chunk_size - chunk_overlap
    start = 0
    while start < len(text):
        pieces.append(text[start : start + chunk_size])
        start += step
    return pieces


def _pack_paragraphs(units: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    """Жадно собирает абзацы в чанки не длиннее ``chunk_size`` с перекрытием."""

    def joined_len(parts: list[str]) -> int:
        return len("\n\n".join(parts)) if parts else 0

    chunks: list[str] = []
    current: list[str] = []
    for unit in units:
        if current and joined_len([*current, unit]) > chunk_size:
            chunks.append("\n\n".join(current))
            # Перекрытие: хвостовые абзацы предыдущего чанка переносим в начало нового.
            overlap: list[str] = []
            for prev in reversed(current):
                if joined_len([prev, *overlap]) > chunk_overlap:
                    break
                overlap.insert(0, prev)
            # Перекрытие не должно вытеснить сам новый абзац за пределы бюджета.
            while overlap and joined_len([*overlap, unit]) > chunk_size:
                overlap.pop(0)
            current = overlap
        current.append(unit)
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def split_markdown(
    markdown: str,
    *,
    source: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Chunk]:
    """Разбивает markdown-документ на чанки.

    Args:
        markdown: исходный текст документа.
        source: имя файла-источника (попадёт в payload Qdrant).
        chunk_size: максимальная длина чанка в символах.
        chunk_overlap: желаемое перекрытие соседних чанков в символах.

    Returns:
        Список чанков со сквозной нумерацией внутри документа.

    Raises:
        ValueError: если ``chunk_overlap >= chunk_size`` — иначе нарезка
            не сможет продвигаться вперёд (классический бесконечный цикл).
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap должен быть меньше chunk_size")

    title = extract_title(markdown, fallback=source)
    pieces: list[str] = []
    for section in _split_sections(markdown):
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", section) if p.strip()]
        units: list[str] = []
        for paragraph in paragraphs:
            if len(paragraph) > chunk_size:
                units.extend(_hard_split(paragraph, chunk_size, chunk_overlap))
            else:
                units.append(paragraph)
        pieces.extend(_pack_paragraphs(units, chunk_size, chunk_overlap))

    return [
        Chunk(text=piece, source=source, title=title, chunk_index=index)
        for index, piece in enumerate(pieces)
    ]
