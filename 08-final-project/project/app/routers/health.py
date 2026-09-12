"""Эндпоинты живости и готовности.

/health (liveness)  — «процесс жив и отвечает». Никаких внешних проверок:
    иначе падение Qdrant заставит оркестратор перезапускать здоровый процесс.
/ready (readiness)  — «готов обслуживать»: проверяем Qdrant и Ollama.
    Балансировщик/оркестратор по этому статусу решает, слать ли трафик.
"""

from __future__ import annotations

from fastapi import APIRouter, Request, Response, status

from app import __version__
from app.schemas import DependencyStatus, HealthResponse, ReadyResponse

router = APIRouter(tags=["service"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness-проба: всегда 200, если процесс в состоянии ответить."""
    return HealthResponse(version=__version__)


@router.get(
    "/ready",
    response_model=ReadyResponse,
    responses={
        503: {
            "model": ReadyResponse,
            "description": "Деградация: то же тело, в checks видно, кто именно не отвечает",
        }
    },
)
async def ready(request: Request, response: Response) -> ReadyResponse:
    """Readiness-проба: доступны ли внешние зависимости.

    При деградации возвращаем 503 — стандартный сигнал «не слать трафик»,
    но в теле честно перечисляем, что именно отвалилось. Статус тела
    вычисляет схема (``ReadyResponse.from_checks``): роутер не может
    ошибиться и вернуть ready при упавшем чеке.
    """
    checks: list[DependencyStatus] = []
    vectorstore = getattr(request.app.state, "vectorstore", None)
    llm = getattr(request.app.state, "llm", None)

    if vectorstore is None or llm is None:
        checks.append(
            DependencyStatus(
                name="services", ok=False, detail="сервисы не инициализированы"
            )
        )
    else:
        qdrant_ok = await vectorstore.healthy()
        checks.append(
            DependencyStatus(
                name="qdrant",
                ok=qdrant_ok,
                detail=None
                if qdrant_ok
                else "недоступен или нет коллекции (запустите индексацию)",
            )
        )
        llm_ok = await llm.healthy()
        checks.append(
            DependencyStatus(
                name="ollama",
                ok=llm_ok,
                detail=None if llm_ok else "сервер не отвечает",
            )
        )

    body = ReadyResponse.from_checks(checks)
    if body.status != "ready":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return body
