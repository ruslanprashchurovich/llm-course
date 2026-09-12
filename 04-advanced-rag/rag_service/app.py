"""FastAPI-обёртка над пайплайном: /ask, /healthz, /metrics.

Запуск (из корня модуля, с поднятым Qdrant и проиндексированной базой):

    uvicorn rag_service.app:app --host 127.0.0.1 --port 8004

Чем отличается от сервисов модуля 3:
  * авторизация: заголовок X-API-Token -> Principal, права уходят в фильтр поиска;
  * /metrics в формате Prometheus — метрики копятся в observability.py;
  * trace_id на каждый запрос: он в логах, в ответе и в заголовке X-Trace-Id.

Эндпоинты объявлены обычными def (не async): пайплайн синхронный, и FastAPI
сам уносит такие обработчики в пул потоков — event loop не блокируется
(та самая лесенка из урока 3.2 здесь не воспроизводится).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .access import DEMO_PRINCIPALS, Principal
from .config import settings
from .llm import OllamaLLM
from .observability import (
    CONTENT_TYPE_LATEST,
    configure_logging,
    log_event,
    metrics_snapshot,
    new_trace_id,
)
from .pipeline import RagPipeline

# ---------------------------------------------------------------------------
# Модели запроса/ответа: Pydantic — граница валидации (урок 3.2)
# ---------------------------------------------------------------------------


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int | None = Field(
        default=0, ge=0, le=10, description="0 = значение из конфига"
    )
    model_config = ConfigDict(extra="forbid")

    @field_validator("question")
    @classmethod
    def not_blank(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 3:
            raise ValueError("вопрос слишком короткий")
        return v


class SourceOut(BaseModel):
    n: int
    doc_id: str
    title: str
    updated_at: str | None = None
    source_type: str | None = None
    score: float


class AskResponse(BaseModel):
    answer: str
    outcome: str
    trace_id: str
    sources: list[SourceOut]


# ---------------------------------------------------------------------------
# Авторизация: токен -> Principal
# ---------------------------------------------------------------------------


def get_principal(x_api_token: str | None = Header(default=None)) -> Principal:
    """Демо-схема: статические токены. В проде — JWT/OIDC и claims из IdP."""
    if x_api_token is None:
        raise HTTPException(status_code=401, detail="нет заголовка X-API-Token")
    principal = DEMO_PRINCIPALS.get(x_api_token)
    if principal is None:
        raise HTTPException(status_code=401, detail="неизвестный токен")
    return principal


# ---------------------------------------------------------------------------
# Приложение
# ---------------------------------------------------------------------------


def create_app(pipeline: RagPipeline | None = None) -> FastAPI:
    """Фабрика приложения. pipeline можно подменить в тестах (DI, урок 3.8)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging(settings.log_level, service=settings.service_name)
        app.state.pipeline = pipeline or RagPipeline()
        if app.state.pipeline.lexical is None and app.state.pipeline.kb.available():
            try:
                app.state.pipeline.build_lexical_index()
            except Exception:  # noqa: BLE001 - коллекции может ещё не быть
                log_event("app.lexical_index_skipped", level=30)
        log_event("app.started", collection=settings.collection)
        yield
        log_event("app.stopped")

    app = FastAPI(title="Vectorika RAG (модуль 4)", lifespan=lifespan)

    @app.middleware("http")
    async def trace_middleware(request: Request, call_next):  # noqa: ANN202
        trace_id = new_trace_id()
        response = await call_next(request)
        response.headers["X-Trace-Id"] = trace_id
        return response

    # -- служебные эндпоинты -------------------------------------------
    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        pipe: RagPipeline = app.state.pipeline
        qdrant_ok = pipe.kb.available()
        llm = pipe.llm
        llm_ok = bool(getattr(llm, "available", lambda: False)())
        is_stub = not isinstance(llm, OllamaLLM)
        status = "ok" if (qdrant_ok and llm_ok and not is_stub) else "degraded"
        return {
            "status": status,
            "qdrant": qdrant_ok,
            "llm": llm_ok,
            "llm_is_stub": is_stub,
            "collection": settings.collection,
        }

    @app.get("/metrics")
    def metrics() -> Response:
        # Prometheus раз в scrape_interval делает GET /metrics и забирает
        # текущие значения счётчиков — сервис ничего никуда не «шлёт» сам.
        return Response(content=metrics_snapshot(), media_type=CONTENT_TYPE_LATEST)

    # -- основной эндпоинт ---------------------------------------------
    @app.post("/ask", response_model=AskResponse)
    def ask(
        body: AskRequest, principal: Principal = Depends(get_principal)
    ) -> AskResponse:
        pipe: RagPipeline = app.state.pipeline
        result = pipe.ask(
            body.question,
            principal,
            final_k=body.top_k or None,
        )
        return AskResponse(
            answer=result.answer,
            outcome=result.outcome,
            trace_id=result.trace_id,
            sources=[SourceOut(**source) for source in result.sources],
        )

    return app


app = create_app()

__all__ = ["create_app", "app"]
