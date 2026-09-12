"""Судья, которому нельзя заговорить зубы: тесты в отдельном процессе (урок 7.2).

Код, написанный LLM, — недоверенный (урок 4.2): гоняем его в subprocess с
таймаутом. Отчёт тестов — интерфейс для фикс-цикла, поэтому каждая строка
читается: вход -> получили -> ожидалось.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

CODE_BLOCK = re.compile(r"```(?:python)?\s*(.+?)```", re.DOTALL)


def extract_code(text: str) -> str | None:
    """Валидация формата: из ответа агента достаём блок кода (или None)."""
    match = CODE_BLOCK.search(text)
    return match.group(1).strip() if match else None


HARNESS = """
import json
results = []
{code}
for expr, expected in {tests!r}:
    try:
        got = eval(expr)
        results.append({{"case": expr, "ok": got == expected, "got": repr(got), "expected": expected}})
    except Exception as exc:
        results.append({{"case": expr, "ok": False, "got": f"{{type(exc).__name__}}: {{exc}}", "expected": expected}})
for expr in {errors!r}:
    try:
        eval(expr)
        results.append({{"case": expr + " => ValueError", "ok": False, "got": "исключения не было", "expected": "ValueError"}})
    except ValueError:
        results.append({{"case": expr + " => ValueError", "ok": True, "got": "ValueError", "expected": "ValueError"}})
    except Exception as exc:
        results.append({{"case": expr + " => ValueError", "ok": False, "got": f"{{type(exc).__name__}}", "expected": "ValueError"}})
print(json.dumps(results, ensure_ascii=False))
"""


def run_tests(
    code_str: str, tests: list, error_cases: list, timeout_s: float = 10.0
) -> list[dict]:
    """Гоняем код LLM в ОТДЕЛЬНОМ процессе с таймаутом (недоверенный код!)."""
    source = HARNESS.format(code=code_str, tests=tests, errors=error_cases)
    with tempfile.NamedTemporaryFile(
        "w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(source)
        path = f.name
    try:
        proc = subprocess.run(
            [sys.executable, "-X", "utf8", path],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout_s,
        )
        if proc.returncode != 0:  # код не загрузился (SyntaxError и т.п.)
            return [
                {
                    "case": "<загрузка кода>",
                    "ok": False,
                    "got": proc.stderr.strip()[-200:],
                    "expected": "импорт без ошибок",
                }
            ]
        return json.loads(proc.stdout)
    except subprocess.TimeoutExpired:
        return [
            {
                "case": "<выполнение>",
                "ok": False,
                "got": f"timeout {timeout_s:g} c (вечный цикл?)",
                "expected": "завершение",
            }
        ]
    finally:
        Path(path).unlink(missing_ok=True)


def report_text(results: list[dict]) -> str:
    return "\n".join(
        f"[{'PASS' if r['ok'] else 'FAIL'}] {r['case']} -> {r['got']}"
        f" (ожидалось {r['expected']})"
        for r in results
    )


@dataclass(frozen=True)
class Verdict:
    results: list[dict]
    report: str
    passed: int
    total: int

    @property
    def ok(self) -> bool:
        return self.passed == self.total

    @property
    def first_fail(self) -> dict | None:
        return next((r for r in self.results if not r["ok"]), None)


class TestJudge:
    """Приёмочные тесты одной задачи, упакованные в вызываемый объект."""

    __test__ = False  # pytest: это не тест-класс, а судья

    def __init__(self, tests: list, error_cases: list, timeout_s: float = 10.0) -> None:
        self.tests = list(tests)
        self.error_cases = list(error_cases)
        self.timeout_s = timeout_s

    @property
    def total(self) -> int:
        return len(self.tests) + len(self.error_cases)

    def __call__(self, code_str: str) -> Verdict:
        results = run_tests(code_str, self.tests, self.error_cases, self.timeout_s)
        return Verdict(
            results,
            report_text(results),
            sum(1 for r in results if r["ok"]),
            len(results),
        )

    def calibrate(self, broken_code: str) -> None:
        """Судья обязан провалить заведомо сломанный код — иначе он не судья."""
        verdict = self(broken_code)
        if verdict.ok:
            raise AssertionError("судья пропустил заведомо сломанный код!")
