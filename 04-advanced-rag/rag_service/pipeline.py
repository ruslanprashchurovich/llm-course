"""Полный путь запроса: права -> поиск -> промпт -> LLM -> проверка ответа.

Это «сборочный цех» пакета: каждый рубеж написан в своём модуле
(access, retrieval, prompts, security), здесь они выстроены в конвейер.

Порядок рубежей не случаен (принцип «дешёвое — первым», урок 3.4):
  1. проверка вопроса (regex, микросекунды);
  2. фильтр прав — применяется ВНУТРИ поиска, а не после него;
  3. порог релевантности — отказ ДО обращения к LLM;
  4. укреплённый промпт с canary и nonce;
  5. выходной контроль ответа (canary, чужие ссылки, цитаты, секреты).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .access import Principal
from .config import settings
from .llm import LLMResult, OllamaLLM, StubLLM, get_llm
from .observability import (
    RAG_ANSWER_BLOCKED,
    RAG_CHUNKS_USED,
    RAG_INJECTION_BLOCKED,
    RAG_NO_CONTEXT,
    RAG_REQUESTS,
    RAG_TOP_SIMILARITY,
    log_event,
    new_trace_id,
    stage,
)
from .prompts import REFUSAL_NO_CONTEXT, build_prompt, is_refusal
from .retrieval import LexicalIndex, Reranker, RetrievalResult, retrieve
from .security import (
    SAFE_FALLBACK_ANSWER,
    AnswerVerdict,
    check_answer,
    make_canary,
    scan_injection,
    scrub_for_logs,
)
from .store import Hit, KnowledgeBase


@dataclass
class AskResult:
    """Всё, что нужно знать о запросе: ответ, источники и паспорт проверок."""

    answer: str
    outcome: str  # answered | refused_no_context | rejected_query | blocked_output
    trace_id: str = ""
    hits: list[Hit] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    verdict: AnswerVerdict | None = None
    llm: LLMResult | None = None
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.outcome == "answered"


class RagPipeline:
    """Конвейер вопрос-ответ с эшелонированной обороной.

    Все зависимости можно подменить (инъекция зависимостей, урок 3.8):
    в тестах вместо Ollama — StubLLM, вместо живого Qdrant — заранее
    собранные Hit'ы через подмену kb.
    """

    def __init__(
        self,
        kb: KnowledgeBase | None = None,
        embedder: Any | None = None,
        llm: OllamaLLM | StubLLM | None = None,
        lexical: LexicalIndex | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        from .embeddings import get_embedder

        self.kb = kb or KnowledgeBase()
        self.embedder = embedder or get_embedder()
        self.llm = llm or get_llm()
        self.lexical = lexical
        self.reranker = reranker

    # ------------------------------------------------------------------
    def build_lexical_index(self) -> LexicalIndex:
        """BM25-индекс строится по всей коллекции и живёт в памяти процесса.

        Права доступа при поиске всё равно проверяются (предикат allow),
        но пересобирать индекс нужно после каждой переиндексации.
        """
        self.lexical = LexicalIndex.from_kb(self.kb)
        return self.lexical

    # ------------------------------------------------------------------
    def ask(
        self,
        question: str,
        principal: Principal,
        *,
        hardened: bool = True,
        output_guard: bool = True,
        final_k: int | None = None,
        min_similarity: float | None = None,
        max_risk: str = "medium",
        only_current: bool = True,
        use_bm25: bool | None = None,
        temperature: float = 0.0,
    ) -> AskResult:
        trace_id = new_trace_id()
        log_event(
            "ask.start",
            user_id=principal.user_id,
            tenant=principal.tenant_id,
            question=scrub_for_logs(question, limit=200),
            question_len=len(question),
        )

        # Рубеж 1: вопрос тоже вход — грубые инъекции отбиваем без LLM (урок 3.7)
        with stage("guard.query"):
            query_report = scan_injection(question)
            if query_report.risk == "high":
                for rule in query_report.matched_rules:
                    RAG_INJECTION_BLOCKED.labels(stage="query", rule=rule).inc()
                RAG_REQUESTS.labels(outcome="rejected_query", tenant=principal.tenant_id).inc()
                log_event(
                    "ask.rejected_query",
                    level=logging.WARNING,
                    rules=query_report.matched_rules,
                )
                return AskResult(
                    answer=REFUSAL_NO_CONTEXT,
                    outcome="rejected_query",
                    trace_id=trace_id,
                    stats={"query_risk": query_report.risk},
                )

        # Рубеж 2: поиск с фильтром прав (внутри retrieve) и порогом
        retrieval: RetrievalResult = retrieve(
            question,
            principal,
            self.kb,
            self.embedder,
            lexical=self.lexical,
            reranker=self.reranker,
            llm=self.llm,
            final_k=final_k,
            min_similarity=min_similarity,
            use_bm25=use_bm25,
            max_risk=max_risk,
            only_current=only_current,
        )
        RAG_TOP_SIMILARITY.observe(retrieval.top_similarity)
        RAG_CHUNKS_USED.observe(len(retrieval.hits))

        if not retrieval.hits:
            RAG_NO_CONTEXT.inc()
            RAG_REQUESTS.labels(outcome="refused_no_context", tenant=principal.tenant_id).inc()
            log_event("ask.no_context", stats=retrieval.stats)
            return AskResult(
                answer=REFUSAL_NO_CONTEXT,
                outcome="refused_no_context",
                trace_id=trace_id,
                stats=retrieval.stats,
            )

        # Рубеж 3: промпт с явными границами данных
        canary = make_canary() if hardened else ""
        system, user, mapping, context_text = build_prompt(
            question,
            retrieval.hits,
            hardened=hardened,
            canary=canary,
            max_chars=settings.max_context_chars,
        )

        with stage("llm.generate", model=getattr(self.llm, "model", "?")):
            llm_result = self.llm.chat(system, user, temperature=temperature)

        # Рубеж 4: выходной контроль
        verdict: AnswerVerdict | None = None
        answer = llm_result.text
        # отказ модели своими словами — легитимный исход, ему цитаты не нужны
        outcome = "refused_no_answer" if is_refusal(answer) else "answered"
        if output_guard:
            with stage("guard.answer"):
                verdict = check_answer(
                    answer,
                    allowed_citations=mapping.keys(),
                    context_text=context_text,
                    canary=canary or None,
                    require_citation=hardened and outcome == "answered",
                )
            if not verdict.allowed:
                for problem in verdict.problems:
                    RAG_ANSWER_BLOCKED.labels(reason=problem.split(":")[0]).inc()
                log_event(
                    "ask.blocked_output",
                    level=logging.WARNING,
                    problems=verdict.problems,
                    answer_preview=scrub_for_logs(answer, limit=200),
                )
                answer = SAFE_FALLBACK_ANSWER
                outcome = "blocked_output"

        sources = [
            {
                "n": index,
                "doc_id": hit.doc_id,
                "title": hit.title,
                "updated_at": hit.metadata.get("updated_at"),
                "source_type": hit.metadata.get("source_type"),
                "score": hit.final_score,
            }
            for index, hit in mapping.items()
        ]

        RAG_REQUESTS.labels(outcome=outcome, tenant=principal.tenant_id).inc()
        log_event(
            "ask.done",
            outcome=outcome,
            n_sources=len(sources),
            top_similarity=retrieval.top_similarity,
            llm_ms=round(llm_result.duration_s * 1000, 1),
            completion_tokens=llm_result.completion_tokens,
        )
        return AskResult(
            answer=answer,
            outcome=outcome,
            trace_id=trace_id,
            hits=retrieval.hits,
            sources=sources,
            verdict=verdict,
            llm=llm_result,
            stats={**retrieval.stats, "query_risk": query_report.risk},
        )


__all__ = ["RagPipeline", "AskResult"]
