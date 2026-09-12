"""POST /api/generate — генерация ответа по документации (+ GET /api/history).

Два режима:

* ``stream=false`` — обычный JSON-ответ (проще для скриптов и тестов);
* ``stream=true``  — Server-Sent Events: событие ``sources`` с источниками,
  затем поток ``token``, в конце ``done`` (или ``error``). Каждое событие —
  модель из app/schemas.py; на провод их переводит ``format_sse``.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse

from app.dependencies import get_history, get_rag
from app.metrics import GENERATED_TOKENS, GENERATION_LATENCY, LLM_ERRORS
from app.schemas import (
    DoneEvent,
    ErrorEvent,
    ErrorResponse,
    GenerateRequest,
    GenerateResponse,
    HistoryItem,
    SourceRef,
    SourcesEvent,
    TokenEvent,
    format_sse,
)
from app.services.history import HistoryRepository
from app.services.llm import LLMError
from app.services.rag import RagService
from app.services.vectorstore import ScoredChunk

logger = logging.getLogger(__name__)
router = APIRouter(tags=["rag"])

# Документация второго варианта ответа для OpenAPI: response_model описывает
# JSON, а поток SSE FastAPI сам описать не может — подсказываем примером.
_SSE_DOC: dict[str, Any] = {
    "description": (
        "JSON-ответ при stream=false или поток Server-Sent Events при stream=true: "
        "события sources -> token* -> done | error (см. схемы *Event)"
    ),
    "content": {
        "text/event-stream": {
            "schema": {"type": "string"},
            "example": (
                'event: sources\ndata: {"sources": [{"number": 1, "source": "05-deploy-guide.md", '
                '"title": "Гид по деплою", "chunk_index": 4, "score": 0.85}]}\n\n'
                'event: token\ndata: {"text": "Откат "}\n\n'
                'event: done\ndata: {"took_ms": 1702, "model": "qwen2.5:3b"}\n\n'
            ),
        }
    },
}


def _to_source_refs(chunks: list[ScoredChunk]) -> list[SourceRef]:
    """Нумерует чанки так же, как они пронумерованы в промпте: [1], [2], ..."""
    return [
        SourceRef(
            number=index,
            source=chunk.source,
            title=chunk.title,
            chunk_index=chunk.chunk_index,
            score=chunk.score,
        )
        for index, chunk in enumerate(chunks, start=1)
    ]


@router.post(
    "/generate",
    response_model=GenerateResponse,
    responses={
        200: _SSE_DOC,
        502: {
            "model": ErrorResponse,
            "description": "LLM-сервер недоступен или вернул ошибку",
        },
        503: {"model": ErrorResponse, "description": "Сервисы ещё не инициализированы"},
    },
)
async def generate(
    body: GenerateRequest,
    rag: Annotated[RagService, Depends(get_rag)],
    history: Annotated[HistoryRepository, Depends(get_history)],
) -> Any:
    """Генерирует ответ по документации с цитированием источников.

    При ``stream=true`` возвращает поток SSE (response_model в этом случае
    не применяется — FastAPI не сериализует готовый Response).
    """
    if body.stream:
        return StreamingResponse(
            _stream_events(body, rag, history),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                # Просим reverse-proxy (nginx) не буферизовать поток.
                "X-Accel-Buffering": "no",
            },
        )

    start = time.perf_counter()
    answer, chunks = await rag.answer(
        body.query, top_k=body.top_k, temperature=body.temperature
    )
    took_ms = int((time.perf_counter() - start) * 1000)
    GENERATION_LATENCY.observe(took_ms / 1000)

    sources = _to_source_refs(chunks)
    await history.save(
        question=body.query,
        answer=answer,
        sources=[ref.source for ref in sources],
        model=rag.model_name,
        took_ms=took_ms,
    )
    # cited / dangling_citations / is_refusal схема посчитает сама (computed_field).
    return GenerateResponse(
        answer=answer, sources=sources, model=rag.model_name, took_ms=took_ms
    )


async def _stream_events(
    body: GenerateRequest,
    rag: RagService,
    history: HistoryRepository,
) -> AsyncIterator[str]:
    """Генератор SSE-событий: sources -> token* -> done | error.

    Историю сохраняем в finally: даже если клиент отвалился на середине,
    частичный ответ не потеряется.
    """
    start = time.perf_counter()
    tokens: list[str] = []
    chunks: list[ScoredChunk] = []
    try:
        chunks, iterator = await rag.stream_answer(
            body.query, top_k=body.top_k, temperature=body.temperature
        )
        yield format_sse(SourcesEvent(sources=_to_source_refs(chunks)))
        async for token in iterator:
            tokens.append(token)
            GENERATED_TOKENS.inc()
            yield format_sse(TokenEvent(text=token))
        took_ms = int((time.perf_counter() - start) * 1000)
        GENERATION_LATENCY.observe(took_ms / 1000)
        yield format_sse(DoneEvent(took_ms=took_ms, model=rag.model_name))
    except LLMError as exc:
        # Статус 200 уже ушёл клиенту вместе с заголовками — ошибку можно
        # сообщить только внутри потока, отдельным событием.
        LLM_ERRORS.inc()
        logger.error("Ошибка LLM во время стрима: %s", exc)
        yield format_sse(ErrorEvent(detail=str(exc)))
    finally:
        answer = "".join(tokens)
        if answer:
            took_ms = int((time.perf_counter() - start) * 1000)
            try:
                await history.save(
                    question=body.query,
                    answer=answer,
                    sources=[chunk.source for chunk in chunks],
                    model=rag.model_name,
                    took_ms=took_ms,
                )
            except Exception:  # noqa: BLE001 - не роняем закрытие стрима из-за истории
                logger.exception("Не удалось сохранить историю")


@router.get(
    "/history",
    response_model=list[HistoryItem],
    responses={
        503: {"model": ErrorResponse, "description": "Сервисы ещё не инициализированы"}
    },
)
async def list_history(
    history: Annotated[HistoryRepository, Depends(get_history)],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> list[HistoryItem]:
    """Последние сохранённые пары вопрос-ответ (новые первыми).

    ``created_at`` на проводе — ISO 8601 в UTC («...Z»): строку SQLite
    без таймзоны схема HistoryItem нормализует сама.
    """
    items = await history.list_recent(limit)
    return [HistoryItem(**item) for item in items]
