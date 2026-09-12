"""FastAPI-прокси перед Ollama — финальная практика модуля 5.

DI как в mini-project модуля 3: create_app(client=...) принимает готовый клиент
(в тестах — фейк со счётчиками), а без него создаёт настоящий в lifespan.
Часы (clock) тоже инжектируются: лимиты на токен тестируются без sleep.

Три уровня практики урока 5.7 живут здесь же:
  * GET /models          — модели с диска Ollama с размерами и digest (базовый);
  * лимит на токен       — 429 + Retry-After, app/ratelimit.py (средний);
  * две модели           — healthz решает, /generate маршрутизирует, app/routing.py (капстоун).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Literal

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Response
from pydantic import BaseModel, Field, computed_field

from .config import Settings
from .metrics import ProxyMetrics
from .ollama_client import OllamaBadRequest, OllamaBusy, OllamaClient, OllamaUnavailable
from .ratelimit import RateLimiter
from .routing import ModelRouter

NS = 1_000_000_000


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=8000)
    num_predict: int | None = Field(default=None, ge=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class GenerateResponse(BaseModel):
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    tokens_per_second: float | None
    load_seconds: float
    queue_ms: float
    # Капстоун 5.7: имя запасной модели, если ответ пришёл от неё; иначе null
    degraded_to: str | None = None
    route: Literal["primary", "fallback", "canary"] = "primary"


class ModelInfo(BaseModel):
    name: str
    size_bytes: int
    digest: str
    modified_at: str | None = None
    parameter_size: str | None = None
    quantization: str | None = None
    role: Literal["primary", "fallback"] | None = None

    @computed_field(description="Размер на диске в ГиБ")
    @property
    def size_gb(self) -> float:
        return round(self.size_bytes / 1024**3, 2)


class ModelsResponse(BaseModel):
    models: list[ModelInfo]
    primary: str
    fallback: str | None
    primary_installed: bool


def create_app(
    client=None,
    settings: Settings | None = None,
    clock: Callable[[], float] | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    clock = clock or time.monotonic

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.client = client if client is not None else OllamaClient(settings)
        app.state.metrics = ProxyMetrics(primary_model=settings.model)
        app.state.semaphore = asyncio.Semaphore(settings.max_concurrent)
        app.state.limiter = RateLimiter(settings.rate_limit_per_minute, clock=clock)
        app.state.router = ModelRouter(
            settings.model,
            settings.fallback_model or None,
            baseline_tps=settings.baseline_tps or None,
            gpu_share_floor=settings.degrade_gpu_share_below,
            p50_ratio=settings.degrade_p50_ratio,
        )
        # Проверка digest при старте (урок 5.6): дрейф не валит сервис,
        # но честно виден в healthz как degraded
        app.state.model_drift: str | None = None
        if settings.pinned_digest:
            actual = await app.state.client.model_digest(settings.model)
            if actual and actual != settings.pinned_digest:
                app.state.model_drift = actual
        yield
        if client is None:
            await app.state.client.aclose()

    app = FastAPI(title="ollama-proxy", version="0.2.0", lifespan=lifespan)

    # --------------------------------------------------------------- auth
    def require_token(x_api_token: str = Header(default="")) -> None:
        """У Ollama нет аутентификации (урок 5.3) — прокси добавляет свою."""
        if settings.api_tokens and x_api_token not in settings.api_tokens:
            raise HTTPException(status_code=401, detail="нет или неверный X-API-Token")

    # --------------------------------------------------------------- rate limit
    def enforce_rate_limit(
        response: Response, x_api_token: str = Header(default="")
    ) -> None:
        """N запросов в минуту на токен; auth выключена — общий ключ anonymous.

        Идёт ПОСЛЕ require_token: чужой токен получает 401, а не тратит чей-то лимит.
        Слот списывается до валидации тела — 422 тоже стоит серверу разбора.
        """
        decision = app.state.limiter.check(x_api_token or "anonymous")
        if not decision.allowed:
            app.state.metrics.observe_error("rate_limited")
            raise HTTPException(
                status_code=429,
                detail=f"лимит {decision.limit} запросов в минуту на токен исчерпан",
                headers={"Retry-After": str(decision.retry_after_s)},
            )
        if decision.limit:
            response.headers["X-RateLimit-Limit"] = str(decision.limit)
            response.headers["X-RateLimit-Remaining"] = str(decision.remaining)

    # --------------------------------------------------------------- generate
    @app.post(
        "/generate",
        response_model=GenerateResponse,
        dependencies=[Depends(require_token), Depends(enforce_rate_limit)],
    )
    async def generate(req: GenerateRequest) -> GenerateResponse:
        metrics: ProxyMetrics = app.state.metrics
        router: ModelRouter = app.state.router

        # Потолок num_predict — на стороне сервера, клиенту доверять нельзя (урок 5.6)
        num_predict = min(
            req.num_predict or settings.num_predict_cap, settings.num_predict_cap
        )
        options = {"num_predict": num_predict, "temperature": req.temperature}

        # Очередь: семафор + честный queue_ms (уроки 3.2, 5.6);
        # не дождались слота — быстрый 503 лучше вечного ожидания (урок 5.3)
        t_queue = time.perf_counter()
        try:
            await asyncio.wait_for(
                app.state.semaphore.acquire(), timeout=settings.queue_timeout_s
            )
        except TimeoutError:
            metrics.observe_error("queue_timeout")
            raise HTTPException(
                status_code=503, detail="очередь занята, повторите позже (с джиттером!)"
            )
        queue_ms = (time.perf_counter() - t_queue) * 1000

        # Маршрут выбираем уже внутри слота: решение роутера мог обновить healthz
        pick = router.pick()
        try:
            data = await app.state.client.generate(
                req.prompt, options, model=pick.model
            )
        except OllamaUnavailable as exc:
            metrics.observe_error("upstream_down")
            raise HTTPException(status_code=502, detail=f"Ollama недоступна: {exc}")
        except OllamaBusy as exc:
            metrics.observe_error("upstream_busy")
            raise HTTPException(status_code=503, detail=f"Ollama перегружена: {exc}")
        except OllamaBadRequest as exc:
            metrics.observe_error("upstream_error")
            raise HTTPException(status_code=502, detail=str(exc))
        except httpx.ReadTimeout:
            # Матрица урока 5.3: генерация не уложилась в read-таймаут — 504 БЕЗ retry
            metrics.observe_error("read_timeout")
            raise HTTPException(
                status_code=504,
                detail="генерация не уложилась в таймаут; повтор не делаем",
            )
        finally:
            app.state.semaphore.release()

        metrics.observe_response(data, queue_ms, model=pick.model, route=pick.kind)
        eval_count = data.get("eval_count", 0)
        eval_duration = data.get("eval_duration", 0)
        return GenerateResponse(
            text=data.get("response", ""),
            model=data.get("model", pick.model),
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=eval_count,
            tokens_per_second=eval_count / (eval_duration / NS)
            if eval_duration
            else None,
            load_seconds=data.get("load_duration", 0) / NS,
            queue_ms=round(queue_ms, 1),
            degraded_to=pick.model if pick.kind == "fallback" else None,
            route=pick.kind,
        )

    # --------------------------------------------------------------- models
    @app.get(
        "/models", response_model=ModelsResponse, dependencies=[Depends(require_token)]
    )
    async def models() -> ModelsResponse:
        """Что лежит на диске Ollama (/api/tags) и какие роли назначены прокси.

        Под токеном: список моделей — инвентарь, а не проверка живости. Сразу видно,
        установлена ли запасная модель, — иначе деградация упрётся в 502 «нет модели».
        """
        try:
            raw = await app.state.client.tags()
        except OllamaUnavailable as exc:
            raise HTTPException(status_code=502, detail=f"Ollama недоступна: {exc}")
        except OllamaBadRequest as exc:
            raise HTTPException(status_code=502, detail=str(exc))

        roles = {settings.model: "primary"}
        if settings.fallback_model:
            roles[settings.fallback_model] = "fallback"
        items = [
            ModelInfo(
                name=m.get("name", "?"),
                size_bytes=int(m.get("size", 0)),
                digest=str(m.get("digest", "")),
                modified_at=m.get("modified_at"),
                parameter_size=(m.get("details") or {}).get("parameter_size"),
                quantization=(m.get("details") or {}).get("quantization_level"),
                role=roles.get(m.get("name", "")),
            )
            for m in raw
        ]
        return ModelsResponse(
            models=items,
            primary=settings.model,
            fallback=settings.fallback_model or None,
            primary_installed=any(i.name == settings.model for i in items),
        )

    # --------------------------------------------------------------- healthz
    @app.get("/healthz")
    async def healthz() -> dict:
        """Пирамида урока 5.5: L1 (version) + L2 (ps, доля GPU) + дрейф digest.

        Глубокую пробу (L3) сюда НЕ кладём: healthz дёргают часто,
        а L3 — деструктивная проверка. Здесь же принимается решение о маршруте
        (капстоун 5.7): у healthz есть и доля GPU, и p50 основной модели.
        """
        metrics: ProxyMetrics = app.state.metrics
        router: ModelRouter = app.state.router
        version = await app.state.client.version()
        if version is None:
            metrics.g_up.set(0)
            return {"status": "down", "reason": "Ollama не отвечает на /api/version"}
        metrics.g_up.set(1)

        loaded = await app.state.client.ps()
        gpu_share = None
        for m in loaded:
            if m.get("name") == settings.model:
                gpu_share = m.get("size_vram", 0) / m.get("size", 1)
                metrics.g_gpu_share.set(gpu_share)

        primary_window = metrics.window
        samples = len(primary_window.tps)
        route = router.evaluate(
            gpu_share=gpu_share,
            p50_tps=primary_window.percentile(0.50),
            samples=samples,
        )
        if router.p50_degraded and samples >= router.MIN_SAMPLES:
            # вердикт «медленно» вынесен — дальше судим по свежим канареечным ответам
            metrics.reset_window(settings.model)
        metrics.g_degraded.set(1 if route.degraded_to else 0)

        status, reasons = "ok", []
        if app.state.model_drift:
            status = "degraded"
            reasons.append(
                f"digest модели изменился: {app.state.model_drift[:19]}… "
                "(прогоните бенчмарк урока 5.2)"
            )
        if route.reason:
            status = "degraded"
            tail = (
                f" — новые запросы идут в {route.degraded_to}"
                if route.degraded_to
                else " — запасная модель не задана, работаем на основной"
            )
            reasons.append(route.reason + tail)

        return {
            "status": status,
            "ollama_version": version,
            "model": settings.model,
            "model_loaded": gpu_share is not None,
            "gpu_share": gpu_share,
            "tps_p50": primary_window.percentile(0.50),
            "cold_starts_last10": primary_window.cold_starts(),
            "reasons": reasons,
            "routing": {
                "active_model": route.active_model,
                "degraded_to": route.degraded_to,
                "fallback_model": settings.fallback_model or None,
                "baseline_tps": route.baseline_tps,
                "reason": route.reason,
            },
        }

    # --------------------------------------------------------------- metrics
    @app.get("/metrics")
    async def metrics_endpoint() -> Response:
        payload, content_type = app.state.metrics.render()
        return Response(content=payload, media_type=content_type)

    return app


# Для uvicorn app.main:app (настройки — из окружения)
app = create_app()
