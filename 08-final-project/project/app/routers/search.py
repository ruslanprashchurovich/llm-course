"""POST /api/search — семантический поиск по документации."""

from __future__ import annotations

import time
from typing import Annotated

from fastapi import APIRouter, Depends

from app.dependencies import get_rag
from app.metrics import SEARCH_LATENCY
from app.schemas import ErrorResponse, SearchRequest, SearchResponse, SearchResult
from app.services.rag import RagService

router = APIRouter(tags=["rag"])


@router.post(
    "/search",
    response_model=SearchResponse,
    responses={
        503: {"model": ErrorResponse, "description": "Сервисы ещё не инициализированы"}
    },
)
async def search(
    body: SearchRequest,
    rag: Annotated[RagService, Depends(get_rag)],
) -> SearchResponse:
    """Возвращает top-k релевантных фрагментов документации.

    Пайплайн: эмбеддинг запроса -> ANN-поиск в Qdrant -> reranker (опц.).
    ``score`` — косинусная близость, ``rerank_score`` — оценка cross-encoder;
    это РАЗНЫЕ шкалы, и поле ответа ``ranked_by`` говорит, по какой из них
    отсортирована выдача. Валидация тела (в том числе правило
    fetch_k >= top_k) — целиком в схеме SearchRequest.
    """
    start = time.perf_counter()
    ranked = await rag.search(
        body.query,
        top_k=body.top_k,
        fetch_k=body.fetch_k,
        use_reranker=body.use_reranker,
        source=body.source,
    )
    took_s = time.perf_counter() - start
    SEARCH_LATENCY.observe(took_s)

    results = [
        SearchResult(
            text=item.chunk.text,
            source=item.chunk.source,
            title=item.chunk.title,
            chunk_index=item.chunk.chunk_index,
            score=item.chunk.score,
            rerank_score=item.rerank_score,
        )
        for item in ranked
    ]
    return SearchResponse(query=body.query, results=results, took_ms=int(took_s * 1000))
