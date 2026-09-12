"""Судья-тесты: калибровка, формат отчёта, таймаут, извлечение кода."""

from team.judge import TestJudge, extract_code, report_text, run_tests
from team.tasks import NORMALIZE_PHONE, PARSE_SIZE
from tests.conftest import BROKEN_CODE, GOOD_CODE, PARTIAL_CODE


def test_extract_code_takes_first_block_only():
    text = "Пояснение\n```python\nx = 1\n```\nи ещё\n```\ny = 2\n```"
    assert extract_code(text) == "x = 1"
    assert extract_code("никакого кода тут нет") is None


def test_good_code_passes_everything(judge):
    verdict = judge(GOOD_CODE)
    assert verdict.ok and verdict.passed == verdict.total == NORMALIZE_PHONE.total


def test_partial_code_is_4_of_6_with_readable_report(judge):
    verdict = judge(PARTIAL_CODE)
    assert (verdict.passed, verdict.total) == (4, 6)
    assert verdict.first_fail is not None
    # Отчёт — интерфейс: вход, что получили, что ожидалось (урок 7.2)
    line = verdict.report.splitlines()[0]
    assert line.startswith("[FAIL] normalize_phone(") and "(ожидалось 79123456789)" in line


def test_calibration_rejects_broken_code(judge):
    judge.calibrate(BROKEN_CODE)          # не бросает — судья ловит брак
    verdict = judge(BROKEN_CODE)
    assert not verdict.ok and verdict.passed == 0


def test_syntax_error_is_reported_as_load_failure():
    results = run_tests("def normalize_phone(text:\n    return", NORMALIZE_PHONE.tests, [])
    assert results == [results[0]] and results[0]["case"] == "<загрузка кода>"
    assert not results[0]["ok"] and "SyntaxError" in results[0]["got"]


def test_infinite_loop_hits_timeout():
    results = run_tests("def parse_size(text):\n    while True:\n        pass",
                        PARSE_SIZE.tests[:1], [], timeout_s=2.0)
    assert results[0]["case"] == "<выполнение>" and "timeout" in results[0]["got"]


def test_error_cases_expect_value_error():
    no_raise = "def normalize_phone(text):\n    return '79123456789'"
    judge = TestJudge([], NORMALIZE_PHONE.error_cases)
    verdict = judge(no_raise)
    assert verdict.passed == 0
    assert all(r["got"] == "исключения не было" for r in verdict.results)
    assert report_text(verdict.results).count("[FAIL]") == 2
