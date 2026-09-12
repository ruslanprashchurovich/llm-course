"""Unit-тесты схем API: контракт проверяется без HTTP — как обычные функции.

Схемы — самый дешёвый слой для тестов: ни клиента, ни приложения, ни фейков.
Здесь фиксируем то, что раньше жило только в головах: строгость разбора,
кросс-полевые правила, производные поля и инварианты ответов.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from app.schemas import (
    DependencyStatus,
    DoneEvent,
    ErrorEvent,
    ErrorResponse,
    GenerateRequest,
    GenerateResponse,
    HistoryItem,
    ReadyResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
    SourceRef,
    SourcesEvent,
    TokenEvent,
    format_sse,
    parse_sse,
)
from app.services.rag import REFUSAL_MARKER, SYSTEM_PROMPT


def error_types(exc_info: pytest.ExceptionInfo[ValidationError]) -> list[str]:
    return [error["type"] for error in exc_info.value.errors()]


def make_ref(number: int, score: float = 0.8) -> SourceRef:
    return SourceRef(
        number=number, source=f"doc-{number}.md", title=f"Документ {number}", chunk_index=0, score=score
    )


# --- строгие запросы -------------------------------------------------------


def test_whitespace_query_is_empty() -> None:
    with pytest.raises(ValidationError) as exc_info:
        SearchRequest(query="   ")
    assert error_types(exc_info) == ["string_too_short"]


def test_query_is_stripped() -> None:
    assert SearchRequest(query="  деплой  ").query == "деплой"


def test_unknown_field_rejected() -> None:
    with pytest.raises(ValidationError) as exc_info:
        SearchRequest.model_validate({"query": "x", "topk": 3})
    assert error_types(exc_info) == ["extra_forbidden"]


def test_no_type_coercion_in_requests() -> None:
    with pytest.raises(ValidationError) as exc_info:
        SearchRequest.model_validate_json('{"query": "x", "top_k": "3"}')
    assert error_types(exc_info) == ["int_type"]
    with pytest.raises(ValidationError) as exc_info:
        GenerateRequest.model_validate_json('{"query": "x", "stream": "yes"}')
    assert error_types(exc_info) == ["bool_type"]


def test_integer_temperature_is_still_a_float() -> None:
    # strict запрещает "0.5" строкой, но целое 0 для float-поля — законный JSON.
    body = GenerateRequest.model_validate_json('{"query": "x", "temperature": 0}')
    assert body.temperature == 0.0


@pytest.mark.parametrize(
    "source", ["../etc/passwd", "docs/05-deploy-guide.md", "05-deploy-guide.txt", ".md"]
)
def test_source_must_be_bare_markdown_filename(source: str) -> None:
    with pytest.raises(ValidationError) as exc_info:
        SearchRequest(query="x", source=source)
    assert error_types(exc_info) == ["string_pattern_mismatch"]


def test_source_accepts_plain_filename() -> None:
    assert SearchRequest(query="x", source=" 05-deploy-guide.md ").source == "05-deploy-guide.md"


def test_fetch_k_not_below_top_k() -> None:
    with pytest.raises(ValidationError) as exc_info:
        SearchRequest(query="x", top_k=5, fetch_k=3)
    assert error_types(exc_info) == ["value_error"]
    assert "fetch_k" in exc_info.value.errors()[0]["msg"]
    assert SearchRequest(query="x", top_k=5, fetch_k=5).fetch_k == 5


def test_fetch_k_requires_reranker() -> None:
    with pytest.raises(ValidationError):
        SearchRequest(query="x", fetch_k=30, use_reranker=False)
    assert SearchRequest(query="x", fetch_k=30).fetch_k == 30


# --- ответы поиска ----------------------------------------------------------


def make_result(score: float, rerank_score: float | None) -> SearchResult:
    return SearchResult(
        text="t", source="a.md", title="A", chunk_index=0, score=score, rerank_score=rerank_score
    )


def test_scores_rounded_only_on_the_wire() -> None:
    result = make_result(0.8700000047683716, -0.60123456)
    dumped = result.model_dump()
    assert dumped["score"] == 0.87
    assert dumped["rerank_score"] == -0.6012
    assert result.score == 0.8700000047683716  # внутри объекта — исходная точность
    assert make_result(0.5, None).model_dump()["rerank_score"] is None


def test_ranked_by_and_count() -> None:
    with_rerank = SearchResponse(query="q", results=[make_result(0.8, -0.6)], took_ms=1)
    without = SearchResponse(query="q", results=[make_result(0.8, None)], took_ms=1)
    empty = SearchResponse(query="q", results=[], took_ms=1)
    assert (with_rerank.ranked_by, with_rerank.count) == ("rerank_score", 1)
    assert (without.ranked_by, without.count) == ("score", 1)
    assert (empty.ranked_by, empty.count) == ("none", 0)
    # computed-поля попадают в JSON наравне с обычными
    assert json.loads(with_rerank.model_dump_json())["ranked_by"] == "rerank_score"


# --- ответы генерации -------------------------------------------------------


def make_answer(answer: str, n_sources: int = 2) -> GenerateResponse:
    return GenerateResponse(
        answer=answer,
        sources=[make_ref(number) for number in range(1, n_sources + 1)],
        model="m",
        took_ms=1,
    )


def test_citations_extracted_sorted_unique() -> None:
    response = make_answer("Ротация важнее вычистки [2]. Секрет отзывается за час [1][2].")
    assert response.cited == [1, 2]
    assert response.dangling_citations == []
    assert response.is_refusal is False


def test_dangling_citation_detected() -> None:
    response = make_answer("Деплой по тегу [1], откат — тем же job [7].")
    assert response.dangling_citations == [7]


def test_refusal_detected_even_with_citation_tail() -> None:
    # Живой ответ qwen2.5:3b из урока 8.4: отказ + хвост из ссылок.
    response = make_answer("В документации ответа не нашёл [1]-[5].", n_sources=5)
    assert response.is_refusal is True
    assert response.cited == [1, 5]
    assert response.dangling_citations == []


def test_refusal_marker_is_shared_with_prompt() -> None:
    assert REFUSAL_MARKER in SYSTEM_PROMPT


def test_numbering_contract_enforced() -> None:
    with pytest.raises(ValidationError) as exc_info:
        GenerateResponse(answer="x", sources=[make_ref(1), make_ref(3)], model="m", took_ms=1)
    assert "пронумерованы 1..2" in exc_info.value.errors()[0]["msg"]


# --- SSE-события ------------------------------------------------------------


def split_sse(wire: str) -> tuple[str, str]:
    event_line, data_line, *_ = wire.split("\n")
    return event_line.removeprefix("event: "), data_line.removeprefix("data: ")


@pytest.mark.parametrize(
    "event",
    [
        SourcesEvent(sources=[make_ref(1, score=0.87)]),
        TokenEvent(text="Деплой "),
        DoneEvent(took_ms=5, model="fake-model"),
        ErrorEvent(detail="Ollama недоступна"),
    ],
    ids=["sources", "token", "done", "error"],
)
def test_sse_roundtrip(event: SourcesEvent | TokenEvent | DoneEvent | ErrorEvent) -> None:
    wire = format_sse(event)
    assert wire.endswith("\n\n")  # разделитель событий
    name, data = split_sse(wire)
    assert "event" not in json.loads(data)  # имя события — только в строке event:
    assert parse_sse(name, data) == event


def test_unknown_sse_event_rejected() -> None:
    with pytest.raises(ValidationError) as exc_info:
        parse_sse("boom", '{"text": "x"}')
    assert error_types(exc_info) == ["union_tag_invalid"]


def test_sse_payload_type_checked() -> None:
    with pytest.raises(ValidationError):
        parse_sse("token", '{"text": 5}')
    with pytest.raises(ValueError):
        parse_sse("token", "[1, 2]")


# --- служебные схемы --------------------------------------------------------


def test_ready_status_is_derived_from_checks() -> None:
    ok = ReadyResponse.from_checks(
        [DependencyStatus(name="qdrant", ok=True), DependencyStatus(name="ollama", ok=True)]
    )
    down = ReadyResponse.from_checks(
        [DependencyStatus(name="qdrant", ok=True), DependencyStatus(name="ollama", ok=False)]
    )
    assert (ok.status, ok.failed) == ("ready", [])
    assert (down.status, down.failed) == ("degraded", ["ollama"])


def test_ready_contradiction_rejected() -> None:
    with pytest.raises(ValidationError):
        ReadyResponse(status="ready", checks=[DependencyStatus(name="ollama", ok=False)])
    with pytest.raises(ValidationError):
        ReadyResponse(status="degraded", checks=[DependencyStatus(name="ollama", ok=True)])


def make_history_item(created_at: str) -> HistoryItem:
    return HistoryItem(
        id=1, created_at=created_at, question="?", answer="!", sources=[], model="m", took_ms=1
    )


def test_history_created_at_becomes_utc_iso() -> None:
    naive = make_history_item("2026-08-28 13:53:12")  # так отдаёт SQLite datetime('now')
    assert naive.model_dump(mode="json")["created_at"] == "2026-08-28T13:53:12Z"
    aware = make_history_item("2026-08-28T13:53:12+03:00")
    assert aware.model_dump(mode="json")["created_at"] == "2026-08-28T10:53:12Z"


def test_error_response_shape() -> None:
    assert ErrorResponse(detail="x").model_dump() == {"detail": "x", "request_id": None}
