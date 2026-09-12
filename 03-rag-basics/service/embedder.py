"""Эмбеддеры за одним интерфейсом.

Контракт (утиная типизация, без ABC — для урока хватит договорённости):

    emb.dim                     -> int, размерность вектора
    emb.fit(texts)              -> обучиться на корпусе (для нейросетевых - no-op)
    emb.embed_docs(texts)       -> list[list[float]], векторы документов
    emb.embed_query(text)       -> list[float], вектор запроса
    emb.save(path) / .load(path)-> состояние на диск и обратно

Qdrant всё равно, откуда взялись числа, — косинус он считает одинаково.
Но правило «одна коллекция — одна модель» (урок 2.1) никуда не делось:
коллекция, набитая TF-IDF-векторами, бесполезна для e5-запросов и наоборот.
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
    """TF-IDF из урока 2.5, упакованный в вектор фиксированной длины.

    Отличие от «нейросетевого» эмбеддера, важное для продакшена:
    у TF-IDF есть СОСТОЯНИЕ, выученное из корпуса (словарь + IDF).
    Запрос обязан кодироваться тем же словарём, что и документы, —
    поэтому состояние сохраняется рядом с индексом и версионируется
    вместе с коллекцией.
    """

    name = "tfidf"

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}  # слово -> номер координаты
        self.idf: list[float] = []

    # --- обучение -------------------------------------------------------
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

    # --- кодирование ----------------------------------------------------
    def _vector(self, text: str) -> list[float]:
        tokens = tokenize(text)
        vec = [0.0] * self.dim
        if not tokens:
            return vec
        for word in tokens:
            i = self.vocab.get(word)  # слов не из словаря просто нет в осях
            if i is not None:
                vec[i] += 1.0
        norm_sq = 0.0
        for i, count in enumerate(vec):
            if count:
                vec[i] = (count / len(tokens)) * self.idf[i]
                norm_sq += vec[i] * vec[i]
        if norm_sq > 0:  # L2-нормировка: косинус = скалярное произведение
            norm = math.sqrt(norm_sq)
            vec = [v / norm for v in vec]
        return vec

    def embed_docs(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    # --- состояние ------------------------------------------------------
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

    Состояния, выученного из НАШЕГО корпуса, у e5 нет — модель обучена
    заранее на чужих терабайтах. Поэтому fit/save — заглушки, а load
    просто создаёт объект. Расплата — ~120 МБ весов и в разы более
    медленное кодирование (на CPU).
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

    def hf_offline(self) -> None:
        """Выключает походы в Hub целиком, а заодно прогресс-бары и отчёты
        загрузки весов: CLI они не нужны. Без оффлайн-режима загрузка модели
        из кэша может зависнуть на минуты, если Hub недоступен по сети."""
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
