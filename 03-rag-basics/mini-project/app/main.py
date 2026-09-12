"""FastAPI-сервис «вопросы к документации» (уроки 3.2, 3.4, 3.5, 3.7).

Прод:    uvicorn app.main:app --host 127.0.0.1 --port 8000
         (перед первым запуском: python -m app.ingest)
Тесты:   create_app(deps=фейки, log_stream=StringIO) - см. tests/conftest.py.

Точка расширения - Deps: три async-функции, за которыми в проде живут
Qdrant и Ollama, а в тестах - фейки. Всё остальное (конвейер, валидация,
метрики, логи) одинаково в обоих мирах - поэтому оно и тестируется.
"""

from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Awaitable, Callable, IO

import structlog
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from .config import Settings, get_settings
from .logs import configure_logging, open_log_file, question_fingerprint
from .metrics import MetricsWindow
from .rag import RagService

log = structlog.get_logger()


@dataclass
class Deps:
    """Внешние зависимости сервиса. Прод - build_real_deps, тесты - фейки."""

    search: Callable[[str, int], Awaitable[list[dict]]]
    generate: Callable[..., Awaitable[str]]      # generate(messages, **options)
    health: Callable[[], Awaitable[dict]]
    aclose: Callable[[], Awaitable[None]] | None = None


def build_real_deps(settings: Settings) -> Deps:
    """Настоящие зависимости: Qdrant + Ollama (ленивые импорты - урок 2.7)."""
    import anyio
    import httpx
    from qdrant_client import AsyncQdrantClient

    from .embedder import make_embedder

    # tfidf загружается из состояния (python -m app.ingest!), e5 - из кэша HF
    embedder = make_embedder(settings.emb_backend, settings.state_path)
    qdrant = AsyncQdrantClient(url=settings.qdrant_url)
    ollama = httpx.AsyncClient(base_url=settings.ollama_url,
                               timeout=httpx.Timeout(120, connect=5))

    async def search(question: str, top_k: int) -> list[dict]:
        if embedder.name == "e5":
            # e5.encode - десятки миллисекунд CPU-работы: в event loop'е
            # это мини-лесенка из урока 3.2 на каждый запрос -> в поток
            vector = await anyio.to_thread.run_sync(embedder.embed_query,
                                                    question)
        else:
            # TF-IDF - микросекунды: дешевле посчитать на месте,
            # чем платить за переключение потока
            vector = embedder.embed_query(question)
        hits = await qdrant.query_points(settings.collection, query=vector,
                                         limit=top_k)
        return [{**hit.payload, "score": hit.score} for hit in hits.points]

    async def generate(messages: list[dict], *, temperature: float = 0.0,
                       num_predict: int | None = None) -> str:
        # options пробрасываются конвейером: retry идёт с t=0.3, self-check
        # и переформулировка - с коротким num_predict (уроки 3.4, 3.5)
        response = await ollama.post("/api/chat", json={
            "model": settings.llm_model, "messages": messages, "stream": False,
            "think": settings.llm_think,        # см. config: thinking-модели
            "options": {"temperature": temperature,
                        "num_predict": num_predict or settings.num_predict,
                        "num_ctx": settings.num_ctx},
        })
        response.raise_for_status()
        return response.json()["message"]["content"]

    async def health() -> dict:
        async def check_qdrant():
            return (await qdrant.count(settings.collection)).count

        async def check_ollama():
            response = await ollama.get("/api/tags", timeout=3)
            return any(m["name"] == settings.llm_model
                       for m in response.json()["models"])

        results = await asyncio.gather(check_qdrant(), check_ollama(),
                                       return_exceptions=True)
        points, model_ready = (
            r if not isinstance(r, Exception) else None for r in results)
        ok = points is not None and model_ready is True
        return {"status": "ok" if ok else "degraded",
                "collection": settings.collection,
                "backend": embedder.name, "points": points,
                "model": settings.llm_model if model_ready else None}

    async def aclose() -> None:
        await qdrant.close()
        await ollama.aclose()

    return Deps(search=search, generate=generate, health=health, aclose=aclose)


# --- модели границы сервиса: Pydantic валидирует форму (урок 3.2) --------------
class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    top_k: int = Field(default=3, ge=1, le=10)


class Source(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    text: str
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    verdict: str          # ok | refused_by_threshold | refused_by_model |
    #                       refused_after_checks | rejected_input
    sanitizer: str        # clean | suspicious | rejected
    checks: dict          # паспорт рубежей (урок 3.4) + переформулировка (3.5)
    search_ms: float
    llm_ms: float
    llm_calls: int


def create_app(deps: Deps | None = None,
               settings: Settings | None = None,
               log_stream: IO[str] | None = None) -> FastAPI:
    settings = settings or get_settings()
    real_mode = deps is None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # логи: тесты приносят свой поток (или конфигурируют structlog сами),
        # прод пишет JSONL в settings.log_path
        if log_stream is not None:
            configure_logging(log_stream)
        elif real_mode:
            configure_logging(open_log_file(settings.log_path))
        app.state.deps = build_real_deps(settings) if real_mode else deps
        app.state.rag = RagService(app.state.deps.search,
                                   app.state.deps.generate,
                                   score_threshold=settings.score_threshold,
                                   guardrails=settings.guardrails,
                                   agent_mode=settings.agent_mode)
        app.state.metrics = MetricsWindow()
        yield
        if app.state.deps.aclose is not None:
            await app.state.deps.aclose()

    app = FastAPI(title="Vectorika Docs Q&A", version="2.0.0",
                  lifespan=lifespan)

    @app.get("/healthz")
    async def healthz(request: Request) -> dict:
        return await request.app.state.deps.health()

    @app.post("/search")
    async def search(request: Request, body: AskRequest) -> list[Source]:
        found = await request.app.state.deps.search(body.question, body.top_k)
        return [Source(**chunk) for chunk in found]

    @app.post("/ask")
    async def ask(request: Request, body: AskRequest) -> AskResponse:
        # request_id привязывается один раз на границе - все события глубже
        # по стеку (rag.py) получают его через contextvars (урок 3.7)
        structlog.contextvars.bind_contextvars(request_id=uuid.uuid4().hex[:8])
        log.info("request_received", **question_fingerprint(body.question))
        try:
            result = await request.app.state.rag.ask(body.question, body.top_k)
            request.app.state.metrics.record(
                result["verdict"],
                search_ms=result["search_ms"], llm_ms=result["llm_ms"])
            log.info("answer_returned", verdict=result["verdict"],
                     llm_calls=result["llm_calls"],
                     search_ms=result["search_ms"], llm_ms=result["llm_ms"])
            return AskResponse(
                answer=result["answer"],
                sources=[Source(**chunk) for chunk in result["sources"]],
                verdict=result["verdict"], sanitizer=result["sanitizer"],
                checks=result["checks"], search_ms=result["search_ms"],
                llm_ms=result["llm_ms"], llm_calls=result["llm_calls"])
        finally:
            structlog.contextvars.clear_contextvars()   # контекст не течёт дальше

    @app.get("/metrics")
    async def metrics(request: Request) -> dict:
        return request.app.state.metrics.summary()

    return app


app = create_app()      # для uvicorn app.main:app
