"""Эмбеддинги для поиска: два бэкенда за одним интерфейсом.

Интерфейс (тот же контракт, что в service/ модуля 3):
    encode_documents(texts) -> list[list[float]]
    encode_query(text)      -> list[float]
    encode_queries(texts)   -> list[list[float]]
    dim                     -> int

Бэкенды:
  * TfidfEmbedder — свой TF-IDF из уроков 2.5/3.2: чистый Python, ноль скачиваний,
    работает сразу. Слабость известна с модуля 2: синонимы («Грузия» vs
    «из-за границы») он не видит — на этом построена часть демонстраций урока 4.1.
  * E5Embedder — intfloat/multilingual-e5-small (~470 МБ с Hugging Face).
    Понимает смысл, но требует скачивания модели. Включается через
    RAG_EMB_BACKEND=e5 — только если модель уже скачана.

Правило из урока 3.2: индекс и запросы считаются ОДНОЙ моделью. Для TF-IDF
это означает, что состояние (словарь + IDF) сохраняется на диск при индексации
и загружается при поиске; для e5 — что имя модели зашито в имя коллекции.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Protocol, Sequence

from .config import settings

TOKEN_RE = re.compile(r"\w+")


class EmbedderProtocol(Protocol):
    """Контракт эмбеддера — то, чего от него ждут ingest и retrieval."""

    @property
    def dim(self) -> int: ...

    def encode_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def encode_query(self, text: str) -> list[float]: ...

    def encode_queries(self, texts: Sequence[str]) -> list[list[float]]: ...


# ---------------------------------------------------------------------------
# TF-IDF: бесплатный бэкенд по умолчанию
# ---------------------------------------------------------------------------


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class TfidfEmbedder:
    """TF-IDF с L2-нормировкой: косинус = скалярное произведение.

    Состояние (vocab + idf) выучивается на корпусе при индексации (fit)
    и обязано версионироваться вместе с коллекцией: запрос кодируется
    ТЕМ ЖЕ словарём, что и документы. Чужой словарь = чужая геометрия
    = мусорные скоры (антипаттерн из урока 4.1).
    """

    def __init__(self, state_path: str | Path | None = None) -> None:
        self.state_path = Path(state_path or settings.tfidf_state_path)
        self.vocab: dict[str, int] = {}
        self.idf: list[float] = []

    # -- обучение и состояние -------------------------------------------
    def fit(self, texts: Sequence[str]) -> "TfidfEmbedder":
        df: dict[str, int] = {}
        for text in texts:
            for word in set(tokenize(text)):
                df[word] = df.get(word, 0) + 1
        self.vocab = {word: i for i, word in enumerate(sorted(df))}
        self.idf = [0.0] * len(self.vocab)
        for word, i in self.vocab.items():
            self.idf[i] = math.log(len(texts) / df[word])
        return self

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path or self.state_path)
        target.write_text(
            json.dumps({"vocab": self.vocab, "idf": self.idf}, ensure_ascii=False),
            encoding="utf-8",
        )
        return target

    def load(self, path: str | Path | None = None) -> "TfidfEmbedder":
        source = Path(path or self.state_path)
        state = json.loads(source.read_text(encoding="utf-8"))
        self.vocab = state["vocab"]
        self.idf = state["idf"]
        return self

    @property
    def is_fitted(self) -> bool:
        return bool(self.vocab)

    # -- интерфейс эмбеддера ---------------------------------------------
    @property
    def dim(self) -> int:
        return len(self.vocab)

    def _vector(self, text: str) -> list[float]:
        if not self.is_fitted:
            raise RuntimeError(
                "TF-IDF не обучен: запустите индексацию (python -m rag_service.ingest) "
                f"или положите состояние в {self.state_path}"
            )
        tokens = tokenize(text)
        vec = [0.0] * self.dim
        if not tokens:
            return vec
        for word in tokens:
            i = self.vocab.get(word)
            if i is not None:
                vec[i] += 1.0
        norm_sq = 0.0
        for i, count in enumerate(vec):
            if count:
                vec[i] = (count / len(tokens)) * self.idf[i]
                norm_sq += vec[i] * vec[i]
        if norm_sq > 0:
            norm = math.sqrt(norm_sq)
            vec = [v / norm for v in vec]
        return vec

    def encode_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def encode_query(self, text: str) -> list[float]:
        return self._vector(text)

    def encode_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]


# ---------------------------------------------------------------------------
# e5: нейроэмбеддер (опционально, требует скачивания)
# ---------------------------------------------------------------------------


class E5Embedder:
    """Ленивая обёртка над SentenceTransformer.

    Модель загружается при первом вызове, а не при импорте. Важная деталь e5:
    модель обучалась с префиксами — документы кодируются как "passage: ...",
    запросы как "query: ...". Забыли префикс — потеряли несколько процентов recall.
    """

    def __init__(self, model_name: str | None = None, device: str = "cpu") -> None:
        self.model_name = model_name or settings.e5_model
        self.device = device
        self._model = None

    @property
    def model(self):  # noqa: ANN201 - тип зависит от установленной библиотеки
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    @property
    def dim(self) -> int:
        return int(self.model.get_embedding_dimension())

    def encode_documents(self, texts: Sequence[str]) -> list[list[float]]:
        prepared = [f"passage: {t}" for t in texts]
        vectors = self.model.encode(
            prepared, batch_size=16, normalize_embeddings=True, convert_to_numpy=True
        )
        return [vector.tolist() for vector in vectors]

    def encode_query(self, text: str) -> list[float]:
        vector = self.model.encode(
            f"query: {text}", normalize_embeddings=True, convert_to_numpy=True
        )
        return vector.tolist()

    def encode_queries(self, texts: Sequence[str]) -> list[list[float]]:
        prepared = [f"query: {t}" for t in texts]
        vectors = self.model.encode(
            prepared, batch_size=8, normalize_embeddings=True, convert_to_numpy=True
        )
        return [vector.tolist() for vector in vectors]


# ---------------------------------------------------------------------------
# Фабрика
# ---------------------------------------------------------------------------

_default_embedder: TfidfEmbedder | E5Embedder | None = None


def get_embedder(
    backend: str | None = None, *, fresh: bool = False
) -> TfidfEmbedder | E5Embedder:
    """Возвращает эмбеддер по настройке RAG_EMB_BACKEND (по умолчанию tfidf).

    Синглтон на процесс: TF-IDF-состояние читается с диска один раз,
    e5 держит одну копию весов в памяти.
    """
    global _default_embedder
    backend = (backend or settings.emb_backend).lower()
    if _default_embedder is not None and not fresh:
        return _default_embedder

    if backend == "e5":
        _default_embedder = E5Embedder()
    else:
        embedder = TfidfEmbedder()
        if embedder.state_path.exists():
            embedder.load()
        _default_embedder = embedder
    return _default_embedder


__all__ = [
    "EmbedderProtocol",
    "TfidfEmbedder",
    "E5Embedder",
    "get_embedder",
    "tokenize",
]
