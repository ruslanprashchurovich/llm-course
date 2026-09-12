"""Роли и промпты команды урока 7.2 — один источник для оркестратора и графа.

Тексты — дословно из урока 7.2. Если поменять формулировку здесь, изменятся
ОБА оркестратора сразу, и тест «журналы совпадают узел в узел» останется честным.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Role:
    key: str           # analyst / developer / tracer
    name: str          # как подписывается в журнале
    system: str
    num_predict: int


ANALYST = Role("analyst", "аналитик",
               "Ты — аналитик. Пиши кратко. НЕ пиши код.", 400)
DEVELOPER = Role("developer", "разработчик",
                 "Ты — Python-разработчик. Возвращай ТОЛЬКО код одним блоком ```python```.",
                 600)
TRACER = Role("tracer", "трассировщик",
              "Ты — код-ревьюер-трассировщик. Работай строго по чеклисту, не переписывай код.",
              400)

ROLES = {role.key: role for role in (ANALYST, DEVELOPER, TRACER)}
ROLE_OF_SYSTEM = {role.system: role.key for role in ROLES.values()}


def analyst_prompt(task: str) -> str:
    return f"Задача:\n{task}\n\nВыпиши 4-6 требований и 3-4 крайних случая списками."


def developer_prompt(task: str, spec: str) -> str:
    return (f"Задача:\n{task}\n\nСпецификация аналитика:\n{spec}\n\n"
            "Напиши функцию. Только код.")


def tracer_prompt(code: str, report: str, first_fail: dict) -> str:
    return (
        f"Код:\n```python\n{code}\n```\n"
        f"Полный отчёт тестов:\n{report}\n\n"
        f"Разбери подробно первый упавший кейс: {first_fail['case']} "
        f"дал {first_fail['got']}, ожидалось {first_fail['expected']}.\n\n"
        "Заполни чеклист, подставляя этот вход ШАГ ЗА ШАГОМ:\n"
        "1. Выпиши значение каждой промежуточной переменной для этого входа.\n"
        "2. На какой строке значение впервые расходится с нужным?\n"
        "3. Причина одним предложением и исправление одним предложением.\n"
        "4. Посмотри на ОСТАЛЬНЫЕ FAIL в отчёте: какие ещё требования "
        "не выполнены? Перечисли каждое одной строкой."
    )


def fix_prompt(task: str, code: str, report: str, trace: str) -> str:
    return (f"Задача:\n{task}\n\nТвой код не прошёл тесты.\n"
            f"```python\n{code}\n```\n\nОтчёт тестов:\n{report}\n\n"
            f"Трассировка ревьюера:\n{trace}\n\n"
            "Верни исправленную функцию целиком. Только код.")


def rebuild_prompt(task: str, report: str, trace: str) -> str:
    """Эскалация: диагноз передаём, старый код ПРЯЧЕМ (анкоринг, урок 7.2)."""
    return (f"Задача:\n{task}\n\nПредыдущая попытка команды провалила тесты:\n"
            f"{report}\n\nДиагноз ревьюера:\n{trace}\n\n"
            "Напиши функцию ЗАНОВО, с чистого листа, учтя диагноз. Только код.")
