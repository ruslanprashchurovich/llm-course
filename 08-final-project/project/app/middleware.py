"""HTTP-middleware: request-id, тайминг, метрики, access-лог."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.metrics import REQUEST_COUNT, REQUEST_LATENCY

logger = logging.getLogger("app.access")


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Сквозной request-id + access-лог + метрики на каждый запрос.

    Замечание: для StreamingResponse время замеряется до отдачи ЗАГОЛОВКОВ,
    а не до конца стрима — тело уходит клиенту уже после возврата ответа
    из middleware. Полное время генерации меряем отдельной метрикой
    (rag_generation_duration_seconds) внутри эндпоинта.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        # request.state живёт в scope запроса: обработчики ошибок (app/main.py)
        # берут отсюда request_id, чтобы положить его в тело ErrorResponse.
        request.state.request_id = request_id
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception(
                "request_id=%s %s %s -> необработанная ошибка",
                request_id,
                request.method,
                request.url.path,
            )
            raise
        elapsed_s = time.perf_counter() - start
        response.headers["X-Request-ID"] = request_id

        # Наши маршруты статичны (без /items/{id}), поэтому сырой path
        # не взорвёт кардинальность меток. С путевыми параметрами сюда
        # нужно подставлять шаблон маршрута — см. урок 5.
        REQUEST_COUNT.labels(
            method=request.method,
            path=request.url.path,
            status=str(response.status_code),
        ).inc()
        REQUEST_LATENCY.labels(method=request.method, path=request.url.path).observe(elapsed_s)

        logger.info(
            "request_id=%s %s %s -> %d за %.0f мс",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            elapsed_s * 1000,
        )
        return response
