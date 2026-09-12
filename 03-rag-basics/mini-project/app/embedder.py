"""Эмбеддеры за одним интерфейсом (урок 3.2): TF-IDF и e5.

Контракт: dim / fit / embed_docs / embed_query / save / load. Qdrant всё
равно, откуда взялись числа, - косинус он считает одинаково. Но правило
«одна коллекция - одна модель» (урок 2.1) никуда не делось: коллекция,
набитая TF-IDF-векторами, бесполезна для e5-запросов и наоборот - поэтому
имя коллекции в config.py получает суффикс бэкенда.

Выбор бэкенда - переменная окружения EMB_BACKEND (см. make_embedder).
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class TfidfEmbedder:
    """TF-IDF из урока 2.5: работает без скачиваний, но со СОСТОЯНИЕМ.

    Словарь + IDF выучиваются из корпуса при индексации и обязаны
    версионироваться вместе с коллекцией: запрос кодируется тем же
    словарём, что и документы.
    """

    name = "tfidf"

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.idf: list[float] = []

    def fit(self, texts: list[str]) -> "TfidfEmbedder":
        df: dict[str, int] = {}
        for text in texts:
            for word in set(tokenize(text)):
                df[word] = df.get(word, 0) + 1
        self.vocab = {word: i for i, word in enumerate(sorted(df))}
        self.idf = [0.0] * len(self.vocab)
        for word, i in self.vocab.items():
            self.idf[i] = math.log(len(texts) / df[word])
        return self

    @property
    def dim(self) -> int:
        return len(self.vocab)

    def _vector(self, text: str) -> list[float]:
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
        if norm_sq > 0:            # L2-нормировка: косинус = скалярное произведение
            norm = math.sqrt(norm_sq)
            vec = [v / norm for v in vec]
        return vec

    def embed_docs(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"vocab": self.vocab, "idf": self.idf}, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "TfidfEmbedder":
        state = json.loads(Path(path).read_text(encoding="utf-8"))
        emb = cls()
        emb.vocab = state["vocab"]
        emb.idf = state["idf"]
        return emb


class E5Embedder:
    """intfloat/multilingual-e5-small (урок 2.1): префиксы обязательны.

    Состояния, выученного из НАШЕГО корпуса, у e5 нет - модель обучена
    заранее на чужих терабайтах. Поэтому fit/save - заглушки, а load
    просто создаёт объект. Расплата - ~120 МБ весов и десятки миллисекунд
    CPU-работы на запрос: в async-обработчике encode обязан уезжать
    в поток (см. build_real_deps в main.py).
    """

    name = "e5"
    MODEL = "intfloat/multilingual-e5-small"
    dim = 384

    def __init__(self) -> None:
        self.hf_offline()
        # ленивый импорт: без sentence-transformers падает только этот бэкенд
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(self.MODEL)

    def fit(self, texts: list[str]) -> "E5Embedder":
        return self

    def embed_docs(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            [f"passage: {t}" for t in texts], normalize_embeddings=True
        )
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(f"query: {text}", normalize_embeddings=True).tolist()

    def save(self, path: str | Path) -> None:
        pass

    @staticmethod
    def hf_offline() -> None:
        """Выключает походы в Hub: модель уже в кэше, а без оффлайн-режима
        загрузка может зависнуть на минуты, если Hub недоступен по сети."""
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

    @classmethod
    def load(cls, path: str | Path) -> "E5Embedder":
        return cls()


BACKENDS = {"tfidf": TfidfEmbedder, "e5": E5Embedder}


def make_embedder(backend: str, state_path: str | Path | None = None):
    """Фабрика: имя бэкенда (+ путь к состоянию) -> готовый эмбеддер."""
    if backend not in BACKENDS:
        raise ValueError(f"неизвестный бэкенд {backend!r}, есть: {sorted(BACKENDS)}")
    cls = BACKENDS[backend]
    if state_path is not None:
        return cls.load(state_path)
    return cls()
