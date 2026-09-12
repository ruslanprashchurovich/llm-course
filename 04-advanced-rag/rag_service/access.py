"""Контроль доступа к документам: кто что имеет право найти.

Главный принцип: фильтр прав применяется НА ЭТАПЕ ПОИСКА (pre-filter), а не
после генерации. Если чанк попал в промпт — считайте, что пользователь его уже
прочитал: модель перескажет содержимое, даже если вы попросите «не показывай».

Аналогия для backend-разработчика: это row-level security. Не «SELECT * а потом
покажем нужное в шаблоне», а «WHERE tenant_id = :tenant AND acl_ok».

Фильтр собирается в синтаксисе Qdrant (урок 2.4): must = логическое И,
FieldCondition поверх payload-индексов. Qdrant применяет условия во время
обхода HNSW — выдача не «дырявится», как у post-фильтра из урока 2.2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

RISK_ORDER = {"low": 0, "medium": 1, "high": 2}


@dataclass(frozen=True)
class Principal:
    """Кто спрашивает. В реальном сервисе собирается из JWT/OIDC-токена."""

    user_id: str
    tenant_id: str = "vectorika"
    # Отделы, документы которых сотруднику видны. "all" — общедоступные документы.
    departments: tuple[str, ...] = ("all",)
    # Уровень допуска: 0 общий, 1 ограниченный, 2 конфиденциальный.
    clearance: int = 0
    roles: tuple[str, ...] = ("employee",)
    # Разрешено ли видеть неперсонализированные (замаскированные) ПДн в ответах.
    may_see_pii: bool = False
    attributes: dict[str, Any] = field(default_factory=dict)

    def can_read(
        self, *, department: str, sensitivity: int, tenant_id: str = "vectorika"
    ) -> bool:
        """Та же логика, что и в фильтре — для юнит-тестов и двойной проверки."""
        return (
            tenant_id == self.tenant_id
            and (department in self.departments or department == "all")
            and sensitivity <= self.clearance
        )


# Демонстрационная «база пользователей»: токен -> Principal.
# В продакшене — валидация JWT и claims из вашего IdP, НИКОГДА не хардкод.
DEMO_PRINCIPALS: dict[str, Principal] = {
    "demo-employee-token": Principal(
        user_id="u-1001",
        departments=("all",),
        clearance=0,
        roles=("employee",),
    ),
    "demo-engineer-token": Principal(
        user_id="u-2002",
        departments=("all", "eng"),
        clearance=1,
        roles=("employee", "engineer"),
    ),
    "demo-hr-token": Principal(
        user_id="u-3003",
        departments=("all", "hr"),
        clearance=2,
        roles=("employee", "hr"),
        may_see_pii=True,
    ),
    "demo-other-tenant-token": Principal(
        user_id="u-9009",
        tenant_id="acme",  # другой арендатор: не должен видеть ничего из vectorika
        departments=("all", "eng"),
        clearance=2,
        roles=("employee",),
    ),
}

ANONYMOUS = Principal(user_id="anonymous", departments=(), clearance=0, roles=())


def build_filter(
    principal: Principal,
    *,
    only_current: bool = True,
    allow_untrusted: bool = True,
    max_risk: str = "medium",
    extra_must: list[Any] | None = None,
) -> Any:
    """Собирает qdrant_client.models.Filter для конкретного пользователя.

    Каждое условие — отсечение по метаданным, записанным при индексации:
      * tenant_id       — граница арендатора (мультитенантность, урок 4.5);
      * sensitivity <=  — уровень допуска;
      * department in   — отделы + общедоступные документы;
      * is_current      — не подсовывать устаревшие редакции (урок 4.1);
      * source_type     — при необходимости только доверенные документы;
      * risk_level <=   — чанки с признаками injection не выше порога (урок 4.2).
    """
    from qdrant_client import models

    must: list[Any] = [
        models.FieldCondition(
            key="tenant_id", match=models.MatchValue(value=principal.tenant_id)
        ),
        models.FieldCondition(
            key="sensitivity", range=models.Range(lte=principal.clearance)
        ),
        models.FieldCondition(
            key="department",
            match=models.MatchAny(
                any=list(dict.fromkeys(("all",) + tuple(principal.departments)))
            ),
        ),
        models.FieldCondition(
            key="risk_level", range=models.Range(lte=RISK_ORDER.get(max_risk, 1))
        ),
    ]
    if only_current:
        must.append(
            models.FieldCondition(key="is_current", match=models.MatchValue(value=True))
        )
    if not allow_untrusted:
        must.append(
            models.FieldCondition(
                key="source_type", match=models.MatchValue(value="trusted")
            )
        )
    if extra_must:
        must.extend(extra_must)
    return models.Filter(must=must)


def describe_filter(flt: Any) -> str:
    """Человекочитаемое представление фильтра — удобно печатать в логах и в уроке."""
    parts: list[str] = []
    for cond in getattr(flt, "must", None) or []:
        key = getattr(cond, "key", "?")
        match = getattr(cond, "match", None)
        rng = getattr(cond, "range", None)
        if match is not None and hasattr(match, "value"):
            parts.append(f"{key} = {match.value}")
        elif match is not None and hasattr(match, "any"):
            parts.append(f"{key} in {list(match.any)}")
        elif rng is not None:
            bounds = []
            if rng.gte is not None:
                bounds.append(f">= {rng.gte}")
            if rng.lte is not None:
                bounds.append(f"<= {rng.lte}")
            parts.append(f"{key} {' и '.join(bounds)}")
        else:
            parts.append(f"{key} ?")
    return " AND ".join(parts) if parts else "(без ограничений)"


__all__ = [
    "Principal",
    "DEMO_PRINCIPALS",
    "ANONYMOUS",
    "build_filter",
    "describe_filter",
    "RISK_ORDER",
]
