"""Точка входа FastAPI-приложения.

Запуск для разработки:
    uvicorn app.main:app --reload

Фабрика ``create_app`` позволяет собирать приложение с любыми настройками —
это ключ к тестируемости (см. tests/conftest.py).
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app
from starlette.exceptions import HTTPException as StarletteHTTPException

from app import __version__
from app.config import Settings, get_settings
from app.logging_setup import setup_logging
from app.middleware import RequestContextMiddleware
from app.routers import generate, health, search
from app.schemas import ErrorResponse
from app.services.llm import LLMError

logger = logging.getLogger(__name__)


async def _init_services(app: FastAPI, settings: Settings) -> None:
    """Создаёт и прогревает все сервисы. Вызывается один раз при старте."""
    # Ленивые импорты: torch и sentence-transformers загружаются секундами
    # и в тестовом окружении не нужны вовсе.
    from app.services.embeddings import EmbeddingService
    from app.services.history import HistoryRepository
    from app.services.llm import OllamaClient
    from app.services.rag import RagService
    from app.services.reranker import RerankerService
    from app.services.vectorstore import VectorStore

    logger.info("Инициализация сервисов (первый запуск скачивает модели)...")

    embeddings = EmbeddingService(
        settings.embedding_model, batch_size=settings.embedding_batch_size
    )
    embeddings.load()

    reranker: RerankerService | None = None
    if settings.reranker_enabled:
        reranker = RerankerService(settings.reranker_model)
        reranker.load()

    vectorstore = VectorStore(
        url=settings.qdrant_url,
        collection=settings.qdrant_collection,
        vector_size=settings.embedding_dim,
        timeout_s=settings.qdrant_timeout_s,
    )
    llm = OllamaClient(
        base_url=settings.ollama_base_url,
        model=settings.llm_model,
        timeout_s=settings.llm_timeout_s,
        think=settings.llm_think,
        num_ctx=settings.llm_num_ctx,
    )
    history = HistoryRepository(settings.history_db_path)
    await history.init()

    app.state.embeddings = embeddings
    app.state.reranker = reranker
    app.state.vectorstore = vectorstore
    app.state.llm = llm
    app.state.history = history
    app.state.rag = RagService(
        settings=settings,
        embeddings=embeddings,
        vectorstore=vectorstore,
        llm=llm,
        reranker=reranker,
    )
    logger.info("Сервисы инициализированы")


async def _shutdown_services(app: FastAPI) -> None:
    """Аккуратно закрывает сетевые клиенты при остановке."""
    llm = getattr(app.state, "llm", None)
    if llm is not None and hasattr(llm, "close"):
        await llm.close()
    vectorstore = getattr(app.state, "vectorstore", None)
    if vectorstore is not None and hasattr(vectorstore, "close"):
        await vectorstore.close()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Жизненный цикл приложения: до yield — старт, после — остановка.

    В тестовом окружении тяжёлая инициализация пропускается: тесты кладут
    в app.state фейковые сервисы (см. tests/conftest.py).
    """
    settings: Settings = app.state.settings
    if settings.environment != "test":
        await _init_services(app, settings)
    yield
    await _shutdown_services(app)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Фабрика приложения: собирает FastAPI с роутерами и middleware."""
    settings = settings or get_settings()
    setup_logging(settings.log_level)

    app = FastAPI(
        title="Docs Assistant",
        description="AI-ассистент по внутренней документации: RAG-поиск и генерация ответов",
        version=__version__,
        lifespan=lifespan,
    )
    app.state.settings = settings

    app.add_middleware(RequestContextMiddleware)

    app.include_router(health.router)
    app.include_router(search.router, prefix="/api")
    app.include_router(generate.router, prefix="/api")

    # Prometheus-метрики отдельным ASGI-приложением на /metrics.
    app.mount("/metrics", make_asgi_app())

    # Все ошибки, кроме 422, — в одном формате ErrorResponse (см. app/schemas.py):
    # клиенту не нужно угадывать форму тела по коду ответа, а request_id
    # в теле ведёт к строкам лога так же надёжно, как заголовок.
    def _error(request: Request, status_code: int, detail: str) -> JSONResponse:
        body = ErrorResponse(
            detail=detail, request_id=getattr(request.state, "request_id", None)
        )
        return JSONResponse(status_code=status_code, content=body.model_dump())

    @app.exception_handler(LLMError)
    async def llm_error_handler(request: Request, exc: LLMError) -> JSONResponse:
        """LLM недоступна/упала -> 502 Bad Gateway (проблема у зависимости)."""
        logger.error("Ошибка LLM: %s", exc)
        return _error(request, 502, f"Ошибка LLM-сервера: {exc}")

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        """503 из dependencies, 404/405 от роутера — тоже ErrorResponse."""
        return _error(request, exc.status_code, str(exc.detail))

    return app


app = create_app()
