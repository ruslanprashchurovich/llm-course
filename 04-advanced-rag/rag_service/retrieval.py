"""Поиск: плотный вектор + BM25, слияние RRF, reranking, мульти-запрос.

Схема конвейера (сверху вниз — всё уже, но всё точнее):

    вопрос
      ├─ (опц.) переформулировки через LLM        multi-query
      ├─ плотный поиск  (эмбеддинги, top-20)      «похоже по смыслу»
      ├─ лексический BM25 (top-20)                «совпадает дословно»
      ├─ RRF-слияние двух списков                 один список кандидатов
      ├─ (опц.) cross-encoder rerank (20 -> 4)    точная, но дорогая модель
      └─ порог отсечения                          лучше отказ, чем мусор

Полный разбор reranking/RRF/multi-query — урок 4.4; здесь конвейер собран
целиком, чтобы уроки 4.1-4.3 могли включать и выключать его части.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol, Sequence

from .access import RISK_ORDER, Principal, build_filter, describe_filter
from .config import settings
from .llm import LLMResult
from .observability import RAG_ACL_DENIED, log_event, stage
from .store import Hit, KnowledgeBase


class ChatLLM(Protocol):
    def chat(
        self, system: str, user: str, /, *, temperature: float, max_tokens: int
    ) -> LLMResult: ...


# --------------------------------------------------------------------------
# Лексический поиск (BM25)
# --------------------------------------------------------------------------

TOKEN_RE = re.compile(r"[a-zA-Zа-яёА-ЯЁ0-9][a-zA-Zа-яёА-ЯЁ0-9._-]*")


def tokenize(text: str) -> list[str]:
    """Токенизация для BM25.

    Точки и дефисы внутри токена сохраняем сознательно: именно так в индексе
    остаются '10.10.0.53', 'inc-2026-014' и 'billing-api' — то, что плотный
    поиск обычно теряет.
    """
    return [token.lower() for token in TOKEN_RE.findall(text)]


class BM25:
    """BM25 Okapi за 40 строк — в духе TF-IDF из урока 2.5.

    Отличия от TF-IDF, ради которых столько формул:
      * насыщение частоты: k1 ограничивает вклад повторов слова
        (10 упоминаний «отпуск» — не в 10 раз релевантнее одного);
      * нормировка длины: b штрафует длинные документы, где слово
        встретится случайно.
    Константы k1=1.5, b=0.75 — стандартные из литературы.
    """

    def __init__(
        self, corpus_tokens: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75
    ) -> None:
        self.k1 = k1
        self.b = b
        self.doc_freqs: list[Counter[str]] = []
        self.doc_len: list[int] = []
        df: dict[str, int] = {}
        for tokens in corpus_tokens:
            freqs = Counter(tokens)
            self.doc_freqs.append(freqs)
            self.doc_len.append(len(tokens))
            for word in freqs:
                df[word] = df.get(word, 0) + 1
        self.n_docs = len(self.doc_freqs)
        self.avgdl = (sum(self.doc_len) / self.n_docs) if self.n_docs else 0.0
        # +1 под логарифмом — сглаживание Lucene: IDF не уходит в минус
        # для слов, которые есть в большинстве документов
        self.idf = {
            word: math.log((self.n_docs - count + 0.5) / (count + 0.5) + 1)
            for word, count in df.items()
        }

    def get_scores(self, query_tokens: Sequence[str]) -> list[float]:
        scores = [0.0] * self.n_docs
        for word in query_tokens:
            idf = self.idf.get(word)
            if idf is None:
                continue
            for index, freqs in enumerate(self.doc_freqs):
                tf = freqs.get(word, 0)
                if not tf:
                    continue
                norm = 1 - self.b + self.b * self.doc_len[index] / self.avgdl
                scores[index] += idf * tf * (self.k1 + 1) / (tf + self.k1 * norm)
        return scores


class LexicalIndex:
    """BM25-индекс в памяти поверх чанков из Qdrant.

    Годится для десятков тысяч чанков (наш корпус — десятки). Для сотен тысяч
    берите полноценный движок: OpenSearch, PostgreSQL FTS, Qdrant sparse-векторы.
    """

    def __init__(self, hits: Sequence[Hit]) -> None:
        self.hits = list(hits)
        corpus = [tokenize(hit.text) for hit in self.hits]
        self.ready = bool(corpus)
        self._bm25 = BM25(corpus) if self.ready else None

    @classmethod
    def from_kb(cls, kb: KnowledgeBase, flt: Any | None = None) -> "LexicalIndex":
        return cls(kb.fetch_all(flt=flt))

    def search(
        self,
        query: str,
        k: int = 20,
        allow: Callable[[Hit], bool] | None = None,
    ) -> list[Hit]:
        """allow — предикат hit -> bool: тот же контроль доступа, что и в векторном поиске."""
        if not self.ready or self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results: list[Hit] = []
        for index in order:
            if scores[index] <= 0:
                break
            source = self.hits[index]
            if allow is not None and not allow(source):
                RAG_ACL_DENIED.inc()
                continue
            results.append(
                Hit(
                    id=source.id,
                    text=source.text,
                    metadata=source.metadata,
                    bm25_score=round(float(scores[index]), 4),
                    ranks={"bm25": len(results) + 1},
                )
            )
            if len(results) >= k:
                break
        return results


# --------------------------------------------------------------------------
# Слияние списков: Reciprocal Rank Fusion
# --------------------------------------------------------------------------


def rrf_fuse(
    runs: dict[str, Sequence[Hit]], k: int = 60, weights: dict[str, float] | None = None
) -> list[Hit]:
    """Reciprocal Rank Fusion: score = sum(w / (k + rank)).

    Почему по рангам, а не по скорам: косинус 0.71 и BM25 12.4 — величины из
    разных вселенных, нормализовать их честно нельзя. Ранги же сравнимы всегда.
    Константа k=60 — из оригинальной статьи (Cormack et al., 2009); она гасит
    разницу между 1-м и 2-м местом, чтобы один список не забивал другой.
    """
    weights = weights or {}
    merged: dict[str, Hit] = {}
    scores: dict[str, float] = {}

    for run_name, hits in runs.items():
        weight = weights.get(run_name, 1.0)
        for rank, hit in enumerate(hits, start=1):
            target = merged.get(hit.id)
            if target is None:
                target = Hit(
                    id=hit.id,
                    text=hit.text,
                    metadata=hit.metadata,
                    similarity=hit.similarity,
                    bm25_score=hit.bm25_score,
                )
                merged[hit.id] = target
            else:
                if hit.similarity is not None:
                    target.similarity = max(target.similarity or 0.0, hit.similarity)
                if hit.bm25_score is not None:
                    target.bm25_score = max(target.bm25_score or 0.0, hit.bm25_score)
            target.ranks[run_name] = rank
            scores[hit.id] = scores.get(hit.id, 0.0) + weight / (k + rank)

    for chunk_id, score in scores.items():
        merged[chunk_id].rrf_score = round(score, 6)
    return sorted(merged.values(), key=lambda h: h.rrf_score or 0.0, reverse=True)


# --------------------------------------------------------------------------
# Reranking cross-encoder'ом (урок 4.4; модель качается отдельно)
# --------------------------------------------------------------------------


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class Reranker:
    """Cross-encoder: читает пару (запрос, чанк) целиком и выдаёт оценку.

    Разница с bi-encoder (обычными эмбеддингами), урок 2.6:
      bi-encoder    — вектор запроса и вектор документа считаются НЕЗАВИСИМО,
                      сравнение — косинус. Дёшево, можно проиндексировать заранее.
      cross-encoder — запрос и документ идут в модель ВМЕСТЕ, attention видит
                      их одновременно. Точнее, но заранее ничего не посчитать:
                      N кандидатов = N прогонов модели.

    Модель (~470 МБ) НЕ качается автоматически — команда скачивания и разбор
    в уроке 4.4.
    """

    def __init__(
        self, model_name: str | None = None, device: str = "cpu", max_length: int = 512
    ) -> None:
        self.model_name = model_name or settings.reranker_model
        self.device = device
        self.max_length = max_length
        self._model = None

    @property
    def model(self):  # noqa: ANN201
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self.model_name, max_length=self.max_length, device=self.device
            )
        return self._model

    def rerank(
        self, query: str, hits: Sequence[Hit], top_n: int = 4, batch_size: int = 8
    ) -> list[Hit]:
        if not hits:
            return []
        pairs = [(query, hit.text) for hit in hits]
        scores = self.model.predict(
            pairs, batch_size=batch_size, show_progress_bar=False
        )
        for hit, score in zip(hits, scores):
            # это логит, а не вероятность: сравнивать можно внутри одного запроса
            hit.rerank_score = round(float(score), 4)
        ordered = sorted(hits, key=lambda h: h.rerank_score or -1e9, reverse=True)
        for rank, hit in enumerate(ordered, start=1):
            hit.ranks["rerank"] = rank
        return ordered[:top_n]


class LLMReranker:
    """Реранкер на локальной LLM: судья оценивает пару (вопрос, чанк).

    Зачем, если есть cross-encoder: cross-encoder точнее и в разы быстрее, но
    требует скачивания модели (~470 МБ). LLM-судья работает на той же qwen, что
    уже поднята для генерации, — реранкинг «бесплатно» с точки зрения зависимостей,
    но платно с точки зрения времени (по одному вызову LLM на кандидата!).
    Годится для десятков кандидатов на учебном стенде; в проде — cross-encoder.

    Интерфейс совпадает с `Reranker.rerank`, поэтому оба взаимозаменяемы.
    Оценка 0..3: 0 не по теме, 1 косвенно, 2 частично, 3 прямой ответ.
    """

    SYSTEM = "Ты оцениваешь релевантность фрагмента документа вопросу сотрудника."

    def __init__(self, llm: ChatLLM, max_chars: int = 400) -> None:
        self.llm = llm
        self.max_chars = max_chars

    def _score_one(self, query: str, text: str) -> int:
        prompt = (
            f"Вопрос сотрудника: {query}\n\n"
            f"Фрагмент документа:\n{text[: self.max_chars]}\n\n"
            "Насколько фрагмент помогает ответить на вопрос? Ответь ОДНОЙ цифрой: "
            "0 (не по теме), 1 (косвенно), 2 (частично), 3 (прямой ответ)."
        )
        try:
            out = self.llm.chat(self.SYSTEM, prompt, temperature=0.0, max_tokens=4).text
            match = re.search(r"[0-3]", out)
            return int(match.group()) if match else 0
        except Exception:  # noqa: BLE001 - модель недоступна/ответила мусором
            return 0

    def rerank(self, query: str, hits: Sequence[Hit], top_n: int = 4) -> list[Hit]:
        for hit in hits:
            hit.rerank_score = float(self._score_one(query, hit.text))
        ordered = sorted(hits, key=lambda h: h.rerank_score or -1.0, reverse=True)
        for rank, hit in enumerate(ordered, start=1):
            hit.ranks["rerank"] = rank
        return ordered[:top_n]


# --------------------------------------------------------------------------
# Мульти-запрос
# --------------------------------------------------------------------------


_LIST_PREFIX = re.compile(r"^\s*(?:[-–—•*]|\d{1,2}[.)])\s*")


def expand_query(
    question: str, llm: ChatLLM, n: int = 3, max_len: int = 160
) -> list[str]:
    from .prompts import MULTIQUERY_SYSTEM, MULTIQUERY_USER

    original = question.strip()
    variants: list[str] = [original]
    seen: set[str] = {original.casefold()}

    try:
        result = llm.chat(
            MULTIQUERY_SYSTEM,
            MULTIQUERY_USER.format(question=original, n=n),
            temperature=0.3,
            max_tokens=200,
        )
        for line in result.text.splitlines():
            candidate = _LIST_PREFIX.sub("", line).strip()
            if not 5 < len(candidate) <= max_len:
                continue
            if candidate.endswith(":"):  # «Вот варианты:» и прочие преамбулы
                continue
            key = candidate.casefold()
            if key in seen:
                continue
            variants.append(candidate)
            seen.add(key)
            if len(variants) > n:
                break
    except Exception as exc:  # noqa: BLE001 - модель недоступна/ответила мусором
        log_event("multiquery.failed", level=30, error=type(exc).__name__)

    return variants[: n + 1]


# --------------------------------------------------------------------------
# Оркестрация
# --------------------------------------------------------------------------


@dataclass
class RetrievalResult:
    hits: list[Hit] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def top_similarity(self) -> float:
        values = [h.similarity for h in self.hits if h.similarity is not None]
        return max(values) if values else 0.0


def retrieve(
    question: str,
    principal: Principal,
    kb: KnowledgeBase,
    embedder: Any,
    *,
    lexical: LexicalIndex | None = None,
    reranker: Reranker | None = None,
    llm: ChatLLM | None = None,
    candidates_k: int | None = None,
    final_k: int | None = None,
    min_similarity: float | None = None,
    use_bm25: bool | None = None,
    use_rerank: bool | None = None,
    use_multiquery: bool | None = None,
    max_risk: str = "medium",
    only_current: bool = True,
) -> RetrievalResult:
    """Полный поиск с правами доступа, слиянием и (опционально) реранкингом."""
    candidates_k = candidates_k or settings.candidates_k
    final_k = final_k or settings.final_k
    min_similarity = (
        settings.min_similarity if min_similarity is None else min_similarity
    )
    use_bm25 = settings.enable_bm25 if use_bm25 is None else use_bm25
    use_rerank = settings.enable_rerank if use_rerank is None else use_rerank
    use_multiquery = (
        settings.enable_multiquery if use_multiquery is None else use_multiquery
    )

    flt = build_filter(principal, only_current=only_current, max_risk=max_risk)
    result = RetrievalResult(queries=[question])
    result.stats["filter"] = describe_filter(flt)

    if use_multiquery and llm is not None:
        with stage("retrieve.multiquery"):
            result.queries = expand_query(question, llm)

    runs: dict[str, Sequence[Hit]] = {}

    with stage("retrieve.dense", queries=len(result.queries), k=candidates_k) as span:
        dense_hits: list[Hit] = []
        seen: set[str] = set()
        for query_index, query in enumerate(result.queries):
            embedding = embedder.encode_query(query)
            # каждая переформулировка — отдельный «прогон» (run) для RRF:
            # ранги внутри прогона должны считаться независимо
            hits_for_query = kb.search(embedding, k=candidates_k, flt=flt)
            runs[f"dense{query_index}"] = hits_for_query
            for hit in hits_for_query:
                if hit.id not in seen:
                    seen.add(hit.id)
                    dense_hits.append(hit)
        span.set(hits=len(dense_hits), unique=len(seen))

    if use_bm25 and lexical is not None:
        with stage("retrieve.bm25") as span:
            # ВАЖНО: BM25 ищет в памяти по всей коллекции, поэтому те же
            # ограничения, что и у dense-фильтра (права, свежесть И risk_level),
            # надо продублировать здесь — иначе лексический путь станет дырой
            # в обороне (отравленный чанк просочится мимо max_risk).
            risk_cap = RISK_ORDER.get(max_risk, 1)

            def _allow(hit: Hit) -> bool:
                meta = hit.metadata
                return (
                    principal.can_read(
                        department=str(meta.get("department", "all")),
                        sensitivity=int(meta.get("sensitivity", 0)),
                        tenant_id=str(meta.get("tenant_id", principal.tenant_id)),
                    )
                    and (not only_current or bool(meta.get("is_current", True)))
                    and int(meta.get("risk_level", 0)) <= risk_cap
                )

            lexical_hits = lexical.search(question, k=candidates_k, allow=_allow)
            runs["bm25"] = lexical_hits
            span.set(hits=len(lexical_hits))

    with stage("retrieve.fuse") as span:
        fused = rrf_fuse(runs) if len(runs) > 1 else list(next(iter(runs.values()), []))
        span.set(candidates=len(fused))

    # порог по косинусу применяем ДО реранкинга: незачем гонять дорогую модель
    # по заведомому мусору. Чанки, найденные только BM25, порог не отсекает.
    filtered = [
        hit
        for hit in fused
        if hit.similarity is None or hit.similarity >= min_similarity or hit.bm25_score
    ]
    result.stats["n_dense"] = len(runs.get("dense0", []))
    result.stats["n_bm25"] = len(runs.get("bm25", []))
    result.stats["n_fused"] = len(fused)
    result.stats["n_after_threshold"] = len(filtered)

    if use_rerank and reranker is not None and filtered:
        with stage("retrieve.rerank", candidates=len(filtered)) as span:
            top = reranker.rerank(
                question, filtered[: max(candidates_k, final_k)], top_n=final_k
            )
            span.set(kept=len(top))
    else:
        top = filtered[:final_k]

    result.hits = top
    result.stats["n_final"] = len(top)
    result.stats["top_similarity"] = result.top_similarity
    return result


def format_hits(hits: Iterable[Hit]) -> str:
    """Компактная таблица результатов — для отладки в ноутбуке."""
    lines = [
        f"{'ранг':<5}{'doc_id':<22}{'sim':>7}{'bm25':>8}{'rrf':>9}{'rerank':>9}  текст"
    ]
    for rank, hit in enumerate(hits, start=1):
        lines.append(
            f"{rank:<5}{hit.doc_id[:20]:<22}"
            f"{(hit.similarity if hit.similarity is not None else float('nan')):>7.3f}"
            f"{(hit.bm25_score or 0):>8.2f}"
            f"{(hit.rrf_score or 0):>9.5f}"
            f"{(hit.rerank_score if hit.rerank_score is not None else float('nan')):>9.3f}"
            f"  {hit.short(60)}"
        )
    return "\n".join(lines)


__all__ = [
    "BM25",
    "LexicalIndex",
    "tokenize",
    "rrf_fuse",
    "Reranker",
    "LLMReranker",
    "sigmoid",
    "expand_query",
    "retrieve",
    "RetrievalResult",
    "format_hits",
]
