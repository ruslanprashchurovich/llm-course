"""База знаний: загрузка документов и чанкинг (компактно из модуля 2).

Проект самодостаточен: не зависит от utils.py модуля — только от папки
с .txt-файлами (первая строка файла = заголовок документа).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str


def load_documents(data_dir: str | Path) -> list[Document]:
    documents = []
    for path in sorted(Path(data_dir).glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        doc_id = re.sub(r"^\d+-", "", path.stem)       # "01-otpusk" -> "otpusk"
        title = text.splitlines()[0].strip()
        documents.append(Document(doc_id=doc_id, title=title, text=text))
    return documents


def chunk_paragraphs(text: str, max_chars: int = 500) -> list[str]:
    """Абзацный чанкинг (урок 2.5): абзацы склеиваются, пока влезают."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = paragraph
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def build_chunks(data_dir: str | Path, max_chars: int = 500) -> list[dict]:
    """Документы -> чанки с контекстными заголовками (уроки 2.5-2.6)."""
    chunks = []
    for doc in load_documents(data_dir):
        for i, piece in enumerate(chunk_paragraphs(doc.text, max_chars)):
            chunks.append({
                "chunk_id": f"{doc.doc_id}_{i}",
                "doc_id": doc.doc_id,
                "title": doc.title,
                "text": piece,
                "indexed_text": piece if i == 0 else f"{doc.title}. {piece}",
            })
    return chunks
