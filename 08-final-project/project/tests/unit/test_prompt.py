"""Unit-тесты сборки промпта: нумерация источников и структура messages."""

from __future__ import annotations

from app.services.rag import SYSTEM_PROMPT, build_context, build_messages
from app.services.vectorstore import ScoredChunk


def make_chunk(number: int) -> ScoredChunk:
    return ScoredChunk(
        text=f"Содержимое фрагмента номер {number}.",
        source=f"doc-{number}.md",
        title=f"Документ {number}",
        chunk_index=0,
        score=0.5,
    )


def test_build_context_numbers_sources() -> None:
    context = build_context([make_chunk(1), make_chunk(2)])
    assert "[1] Источник: doc-1.md" in context
    assert "[2] Источник: doc-2.md" in context
    assert context.index("[1]") < context.index("[2]")


def test_build_context_empty() -> None:
    context = build_context([])
    assert "не найдено" in context


def test_build_messages_structure() -> None:
    messages = build_messages("Как откатить релиз?", [make_chunk(1)])
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == SYSTEM_PROMPT
    assert messages[1]["role"] == "user"
    assert "Как откатить релиз?" in messages[1]["content"]
    assert "Содержимое фрагмента номер 1." in messages[1]["content"]


def test_prompt_mentions_citation_format() -> None:
    """Инструкция о ссылках [N] — контракт с клиентом, фиксируем его тестом."""
    assert "[1]" in SYSTEM_PROMPT
