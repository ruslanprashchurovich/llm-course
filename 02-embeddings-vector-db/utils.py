"""Вспомогательные функции для модуля «Embeddings и векторные базы данных».

Используются в уроках 03-07. Никакой магии: загрузка документов из data/,
простые стратегии чанкинга и косинусная близость на numpy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Категории документов базы знаний «Векторики» (по slug из имени файла).
# Используются как метаданные для фильтров в Chroma/Qdrant.
TOPICS: dict[str, str] = {
    "otpusk": "hr",
    "bolnichny": "hr",
    "udalenka": "hr",
    "onboarding": "hr",
    "vpn": "it",
    "oborudovanie": "it",
    "security": "it",
    "code-review": "dev",
    "git-flow": "dev",
    "deploy": "dev",
    "incidents": "dev",
    "testing": "dev",
    "api-guidelines": "dev",
    "meetings": "office",
}


@dataclass
class Document:
    """Один документ базы знаний."""

    doc_id: str  # slug, например "otpusk"
    title: str  # первая строка файла
    text: str  # полный текст (включая заголовок)
    topic: str  # категория: hr / it / dev / office
    source: str  # имя файла, например "01-otpusk.txt"


@dataclass
class Chunk:
    """Фрагмент документа после чанкинга."""

    chunk_id: str  # например "otpusk_0"
    doc_id: str  # id родительского документа
    text: str
    metadata: dict = field(default_factory=dict)


def load_documents(data_dir: str | Path = "data") -> list[Document]:
    """Загружает все .txt из data_dir и определяет тему по имени файла."""
    data_dir = Path(data_dir)
    docs: list[Document] = []
    for path in sorted(data_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        # имя вида "01-otpusk.txt" -> slug "otpusk"
        slug = re.sub(r"^\d+-", "", path.stem)
        title = text.splitlines()[0].strip() if text else path.stem
        docs.append(
            Document(
                doc_id=slug,
                title=title,
                text=text,
                topic=TOPICS.get(slug, "other"),
                source=path.name,
            )
        )
    if not docs:
        raise FileNotFoundError(
            f"В {data_dir.resolve()} нет .txt файлов. "
            "Запустите ноутбук из папки модуля 02-embeddings-vector-db."
        )
    return docs


def chunk_fixed(text: str, chunk_size: int = 400, overlap: int = 80) -> list[str]:
    """Нарезка на куски фиксированной длины (в символах) с перекрытием.

    Простейшая стратегия: скользящее окно. Старается не резать слово
    пополам — откатывается назад до ближайшего пробела.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap должен быть меньше chunk_size")
    text = text.strip()
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        # не резать слово: откатываемся к последнему пробелу в окне
        if end < len(text):
            last_space = text.rfind(" ", start, end)
            if last_space > start:
                end = last_space
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def chunk_paragraphs(text: str, max_chars: int = 800) -> list[str]:
    """Нарезка по абзацам (пустая строка — разделитель).

    Соседние короткие абзацы склеиваются, пока не превысят max_chars.
    Абзац длиннее max_chars дорезается chunk_fixed.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buf = ""
    for p in paragraphs:
        if len(p) > max_chars:
            if buf:
                chunks.append(buf)
                buf = ""
            chunks.extend(chunk_fixed(p, chunk_size=max_chars, overlap=100))
            continue
        candidate = f"{buf}\n\n{p}" if buf else p
        if len(candidate) <= max_chars:
            buf = candidate
        else:
            chunks.append(buf)
            buf = p
    if buf:
        chunks.append(buf)
    return chunks


def split_sentences(text: str) -> list[str]:
    """Наивная разбивка на предложения (достаточно для учебных текстов)."""
    parts = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    return [s.strip() for s in parts if s.strip()]


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Косинусная близость двух векторов."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Матрица косинусных близостей: строки a x строки b."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return a_norm @ b_norm.T


if __name__ == "__main__":
    docs = load_documents(Path(__file__).parent / "data")
    print(f"Загружено документов: {len(docs)}")
    for d in docs[:3]:
        n_fixed = len(chunk_fixed(d.text))
        n_para = len(chunk_paragraphs(d.text))
        print(
            f"- {d.source} [{d.topic}]: {len(d.text)} симв., "
            f"чанков fixed={n_fixed}, para={n_para}"
        )
