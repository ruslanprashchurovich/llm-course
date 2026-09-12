"""Общие фикстуры: код разной степени сломанности и ScriptedLLM без сети.

Сценарии команды собираются из этих кирпичей: «правильный код», «код, который
проходит 4/6» и «проза вместо кода» — ровно те повороты, что были в уроке 7.2.
"""

from __future__ import annotations

import pytest

from team.judge import TestJudge
from team.llm import ScriptedLLM
from team.log import TeamLog
from team.prompts import ROLE_OF_SYSTEM
from team.tasks import NORMALIZE_PHONE

GOOD_CODE = '''
def normalize_phone(text: str) -> str:
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) != 11 or digits[0] not in "78":
        raise ValueError(f"не похоже на номер: {text!r}")
    return "7" + digits[1:]
'''

# Оставляет скобки (модальный ответ 3B из урока 7.2): ровно 4/6 на приёмке
PARTIAL_CODE = '''
import re
def normalize_phone(text: str) -> str:
    digits = re.sub(r"[^0-9()]", "", text)
    if len(digits) != 11 or digits[0] not in "78":
        raise ValueError("bad")
    return "7" + digits[1:]
'''

BROKEN_CODE = "def normalize_phone(text):\n    return '0'"

SPEC = "Требования:\n1. 11 цифр\n2. первая 7\nКрайние случаи: пусто, буквы"
TRACE = "1. digits = '(912)...'\n2. строка 3\n3. регэксп оставляет скобки; оставить только цифры\n4. -"


def block(code: str) -> str:
    return f"Вот код:\n```python\n{code}\n```"


@pytest.fixture
def judge() -> TestJudge:
    return TestJudge(NORMALIZE_PHONE.tests, NORMALIZE_PHONE.error_cases, timeout_s=15.0)


@pytest.fixture
def quiet_log() -> TeamLog:
    return TeamLog(verbose=False)


def scripted(developer: list[str], analyst: list[str] | None = None,
             tracer: list[str] | None = None) -> ScriptedLLM:
    """Команда 7.2 по сценарию: реплики разработчика задают поворот сюжета."""
    return ScriptedLLM(
        replies={"analyst": analyst or [SPEC], "developer": developer,
                 "tracer": tracer or [TRACE]},
        role_of=ROLE_OF_SYSTEM,
    )


SCENARIOS = {
    # имя: (реплики разработчика по порядку, ожидаемый вердикт, вызовов LLM)
    "accept_v1": ([block(GOOD_CODE)], "ACCEPTED", 2),
    "accept_after_fix": ([block(PARTIAL_CODE), block(GOOD_CODE)], "ACCEPTED", 4),
    "accept_after_escalation": ([block(PARTIAL_CODE), block(PARTIAL_CODE), block(PARTIAL_CODE),
                                 block(GOOD_CODE)], "ACCEPTED (после эскалации)", 7),
    "rejected_human": ([block(PARTIAL_CODE)], "REJECTED: нужен человек", 11),
    "no_code": (["Функция должна убирать лишние символы."],
                "REJECTED: разработчик не вернул код", 2),
}
