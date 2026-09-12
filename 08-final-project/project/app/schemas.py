"""Pydantic-схемы запросов и ответов API.

Схемы — это контракт сервиса: FastAPI валидирует по ним входящие данные
(возвращая 422 при нарушении) и сериализует ответы. Здесь контракт умеет
больше, чем перечислять поля:

* запросы **строгие**: неизвестное поле, число строкой, пробелы вместо текста —
  422 сразу, а не молчаливая догадка (см. ``ApiRequest``);
* правила, связывающие несколько полей (fetch_k >= top_k), живут
  в ``model_validator`` — рядом с полями, а не в роутере;
* ответы **сами считают производные факты** (``computed_field``): какие номера
  [N] процитированы, есть ли «висячие» ссылки, отказ ли это;
* события SSE — типизированное объединение с дискриминатором ``event``:
  протокол стрима записан кодом, а не только в docstring роутера;
* инварианты ответов (status ``ready`` <=> все чеки ok; источники пронумерованы
  1..N) проверяются в схеме — роутер физически не может вернуть противоречие.

Решение и его цена — в docs/adr/0005-api-contract.md.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    TypeAdapter,
    computed_field,
    field_serializer,
    field_validator,
    model_validator,
)

from app.services.rag import REFUSAL_MARKER

# --------------------------------------------------------------------------
# Переиспользуемые типы полей: одно определение — один смысл во всём API
# --------------------------------------------------------------------------
QueryText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=2000),
    Field(
        description="Текст запроса. Пробелы по краям обрезаются; пустая строка — 422"
    ),
]
TopK = Annotated[
    int,
    Field(
        ge=1,
        le=20,
        description="Сколько результатов вернуть (по умолчанию из настроек)",
    ),
]
FetchK = Annotated[
    int,
    Field(
        ge=1,
        le=100,
        description="Сколько кандидатов взять из Qdrant до reranker (по умолчанию из настроек)",
    ),
]
Temperature = Annotated[
    float,
    Field(
        ge=0.0, le=2.0, description="Температура генерации (по умолчанию из настроек)"
    ),
]
# Имя файла без путей и слэшей: это payload-фильтр Qdrant, а не путь на диске,
# но «../etc/passwd» в запросе — всегда признак того, что клиент что-то перепутал.
SourceName = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=200,
        pattern=r"^[\w][\w.\-]*\.md$",
    ),
    Field(description="Имя файла-источника, например 05-deploy-guide.md (без путей)"),
]


class ApiRequest(BaseModel):
    """База всех тел запросов: строгий разбор вместо «догадливого».

    * ``extra="forbid"`` — опечатка ``topk`` вместо ``top_k`` даёт 422, а не
      молча проигнорированный параметр и «почему-то пять результатов»;
    * ``strict=True`` — ``"3"`` не превращается в 3, ``"yes"`` — в True.
      JSON умеет числа и булевы значения, пусть клиент их и присылает;
    * ``str_strip_whitespace`` — пробелы по краям строк не считаются данными.
    """

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)


# --------------------------------------------------------------------------
# /api/search
# --------------------------------------------------------------------------
class SearchRequest(ApiRequest):
    """Тело запроса семантического поиска."""

    # Конфиг наследуется и дополняется: строгость — от ApiRequest, примеры — свои.
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"query": "как откатить деплой на проде?", "top_k": 3},
                {"query": "правила аппрувов", "top_k": 2, "source": "03-git-flow.md"},
                {
                    "query": "как откатить деплой?",
                    "top_k": 5,
                    "fetch_k": 40,
                    "use_reranker": True,
                },
            ]
        }
    )

    query: QueryText
    top_k: TopK | None = None
    fetch_k: FetchK | None = None
    use_reranker: bool | None = Field(
        default=None,
        description="Принудительно включить/выключить reranker для этого запроса",
    )
    source: SourceName | None = None

    @model_validator(mode="after")
    def _check_candidate_window(self) -> Self:
        """Правила на несколько полей сразу — их не выразить одним Field.

        fetch_k — ширина «сети» для reranker'а (урок 8.3: fetch_k != top_k).
        Сеть уже итога бессмысленна, а сеть без reranker'а — параметр, который
        молча проигнорируется; молчаливое игнорирование — тоже баг контракта.
        """
        if self.fetch_k is not None:
            if self.top_k is not None and self.fetch_k < self.top_k:
                raise ValueError(
                    f"fetch_k ({self.fetch_k}) должен быть не меньше top_k ({self.top_k})"
                )
            if self.use_reranker is False:
                raise ValueError(
                    "fetch_k имеет смысл только вместе с reranker: "
                    "уберите fetch_k или не выключайте use_reranker"
                )
        return self


class SearchResult(BaseModel):
    """Один найденный фрагмент документации."""

    text: str
    source: str = Field(description="Имя файла-источника")
    title: str = Field(description="Заголовок документа")
    chunk_index: int = Field(ge=0)
    score: float = Field(
        description="Косинусная близость из Qdrant (шкала примерно 0..1)"
    )
    rerank_score: float | None = Field(
        default=None,
        description="Оценка cross-encoder (другая шкала! не сравнивать со score)",
    )

    @field_serializer("score", "rerank_score")
    def _round_scores(self, value: float | None) -> float | None:
        """Четырёх знаков клиенту достаточно: 0.8700000047683716 — шум float32, не данные.

        Округляется только представление на проводе; внутри объекта — исходное число.
        """
        return None if value is None else round(value, 4)


class SearchResponse(BaseModel):
    """Ответ эндпоинта поиска."""

    query: str
    results: list[SearchResult]
    took_ms: int = Field(ge=0)

    @computed_field(description="Число результатов")
    @property
    def count(self) -> int:
        return len(self.results)

    @computed_field(
        description=(
            "По какой шкале отсортированы результаты: rerank_score (reranker работал), "
            "score (только косинус) или none (пустая выдача)"
        )
    )
    @property
    def ranked_by(self) -> Literal["rerank_score", "score", "none"]:
        """Снимает главную путаницу урока 8.3 — «две шкалы»: клиент видит, какая правит."""
        if not self.results:
            return "none"
        return "rerank_score" if self.results[0].rerank_score is not None else "score"


# --------------------------------------------------------------------------
# /api/generate
# --------------------------------------------------------------------------
class GenerateRequest(ApiRequest):
    """Тело запроса генерации ответа."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"query": "что делать при коммите секрета в git?"},
                {"query": "как объявляется SEV1?", "stream": True},
            ]
        }
    )

    query: QueryText
    top_k: TopK | None = None
    stream: bool = Field(
        default=False, description="true — ответ придёт потоком Server-Sent Events"
    )
    temperature: Temperature | None = None


class SourceRef(BaseModel):
    """Ссылка на источник, использованный при генерации ответа."""

    number: int = Field(
        ge=1, description="Номер источника в тексте ответа, например [1]"
    )
    source: str
    title: str
    chunk_index: int = Field(ge=0)
    score: float

    @field_serializer("score")
    def _round_score(self, value: float) -> float:
        return round(value, 4)


_CITATION = re.compile(r"\[(\d+)\]")


class GenerateResponse(BaseModel):
    """Ответ генерации (не-стриминговый вариант).

    Три computed-поля — то, что иначе каждый клиент считал бы сам и по-своему.
    """

    answer: str
    sources: list[SourceRef]
    model: str
    took_ms: int = Field(ge=0)

    @computed_field(
        description="Номера источников, на которые ссылается ответ: «[1] … [3]» -> [1, 3]"
    )
    @property
    def cited(self) -> list[int]:
        return sorted({int(number) for number in _CITATION.findall(self.answer)})

    @computed_field(
        description="Ссылки на несуществующие источники — модель «придумала» номер. Пустой список — норма"
    )
    @property
    def dangling_citations(self) -> list[int]:
        known = {ref.number for ref in self.sources}
        return [number for number in self.cited if number not in known]

    @computed_field(
        description="Канонический отказ («В документации ответа не нашёл»), даже с хвостом из ссылок"
    )
    @property
    def is_refusal(self) -> bool:
        """По вхождению, не по равенству: 3B-модель дописывает к отказу «[1]-[5]» (урок 8.4)."""
        return REFUSAL_MARKER.casefold() in self.answer.casefold()

    @model_validator(mode="after")
    def _numbering_contract(self) -> Self:
        """Контракт номеров: sources[i].number == i + 1, как в промпте (урок 8.4).

        Нарушение — наш баг, а не ошибка клиента: пусть падает громко (500),
        чем клиент молча прочитает [2] как второй источник.
        """
        numbers = [ref.number for ref in self.sources]
        if numbers != list(range(1, len(self.sources) + 1)):
            raise ValueError(
                f"источники должны быть пронумерованы 1..{len(self.sources)} "
                f"в порядке промпта, получено {numbers}"
            )
        return self


# --------------------------------------------------------------------------
# SSE-события /api/generate?stream=true — протокол стрима в типах
# --------------------------------------------------------------------------
class SourcesEvent(BaseModel):
    """Первое событие: источники ещё до первого токена."""

    event: Literal["sources"] = "sources"
    sources: list[SourceRef]


class TokenEvent(BaseModel):
    """Фрагмент ответа (один чанк стрима Ollama ~ один токен)."""

    event: Literal["token"] = "token"
    text: str


class DoneEvent(BaseModel):
    """Финал успешного стрима с метаданными."""

    event: Literal["done"] = "done"
    took_ms: int = Field(ge=0)
    model: str


class ErrorEvent(BaseModel):
    """Ошибка LLM внутри потока — статус 200 уже ушёл, сообщить можно только так."""

    event: Literal["error"] = "error"
    detail: str


# Discriminated union: по полю event pydantic сам выбирает модель, а неизвестное
# имя события — ValidationError с перечислением допустимых.
StreamEvent = Annotated[
    SourcesEvent | TokenEvent | DoneEvent | ErrorEvent, Field(discriminator="event")
]
_STREAM_EVENTS: TypeAdapter[SourcesEvent | TokenEvent | DoneEvent | ErrorEvent] = (
    TypeAdapter(StreamEvent)
)


def format_sse(event: SourcesEvent | TokenEvent | DoneEvent | ErrorEvent) -> str:
    """Форматирует одно SSE-событие.

    Имя события уходит в строку ``event:``, поля — в ``data:`` без дублирования
    ``event`` (формат на проводе тот же, что в уроке 8.4). Пустая строка
    (двойной ``\\n``) — обязательный разделитель событий: без неё клиент
    склеит всё в одно «вечное» событие.
    """
    return f"event: {event.event}\ndata: {event.model_dump_json(exclude={'event'})}\n\n"


def parse_sse(
    event_name: str, data: str
) -> SourcesEvent | TokenEvent | DoneEvent | ErrorEvent:
    """Обратная операция для клиентов и тестов: имя + JSON -> типизированное событие.

    Неизвестное имя или поле не того типа — ValidationError здесь,
    а не KeyError где-то в глубине клиентского кода.
    """
    payload = json.loads(data)
    if not isinstance(payload, dict):
        raise ValueError("data SSE-события должна быть JSON-объектом")
    return _STREAM_EVENTS.validate_python({"event": event_name, **payload})


# --------------------------------------------------------------------------
# /api/history
# --------------------------------------------------------------------------
class HistoryItem(BaseModel):
    """Сохранённая пара вопрос-ответ."""

    id: int
    created_at: datetime = Field(
        description="Время сохранения: UTC, ISO 8601 с суффиксом Z"
    )
    question: str
    answer: str
    sources: list[str]
    model: str
    took_ms: int

    @field_validator("created_at")
    @classmethod
    def _as_utc(cls, value: datetime) -> datetime:
        """SQLite ``datetime('now')`` — это UTC без пометки; клиенту отдаём явный Z.

        Строка «2026-08-28 13:53:12» без зоны — приглашение каждому клиенту
        трактовать её по-своему (локальное время? UTC?). datetime с tzinfo
        и ISO 8601 на проводе снимают вопрос.
        """
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)


# --------------------------------------------------------------------------
# Служебные эндпоинты и ошибки
# --------------------------------------------------------------------------
class HealthResponse(BaseModel):
    """Liveness-ответ: процесс жив."""

    status: Literal["ok"] = "ok"
    version: str


class DependencyStatus(BaseModel):
    """Статус одной внешней зависимости."""

    name: str
    ok: bool
    detail: str | None = None


class ReadyResponse(BaseModel):
    """Readiness-ответ: готов ли сервис обрабатывать запросы."""

    status: Literal["ready", "degraded"]
    checks: list[DependencyStatus]

    @computed_field(description="Имена зависимостей, не прошедших проверку")
    @property
    def failed(self) -> list[str]:
        return [check.name for check in self.checks if not check.ok]

    @model_validator(mode="after")
    def _status_matches_checks(self) -> Self:
        """Инвариант: ready <=> все чеки ok. Роутер не может «забыть» сменить статус."""
        all_ok = all(check.ok for check in self.checks)
        if (self.status == "ready") != all_ok:
            raise ValueError(
                f"status={self.status!r} противоречит чекам: не прошли {self.failed or 'никто'}"
            )
        return self

    @classmethod
    def from_checks(cls, checks: list[DependencyStatus]) -> Self:
        """Единственный правильный способ собрать ответ: статус вычисляется, не передаётся."""
        return cls(
            status="ready" if all(check.ok for check in checks) else "degraded",
            checks=checks,
        )


class ErrorResponse(BaseModel):
    """Тело любой ошибки сервиса, кроме 422 (у него стандартный формат FastAPI).

    ``request_id`` дублирует заголовок X-Request-ID: по нему ошибку ищут в логах,
    а тело ответа клиенты сохраняют чаще, чем заголовки.
    """

    detail: str = Field(description="Что случилось, человеческим языком")
    request_id: str | None = Field(
        default=None, description="X-Request-ID запроса — ключ для поиска в логах"
    )
