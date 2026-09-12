"""Задание 2: политика конвейера и подсчёт цены выбора — без AutoGen и без сети."""

import pytest

from team.judge import TestJudge
from team.supervisor import (
    APPROVE,
    REWORK,
    SelectionRecord,
    expected_next,
    last_code_block,
    score_selection,
    summarize,
    code_critic_reply,
)
from team.tasks import NORMALIZE_PHONE
from tests.conftest import GOOD_CODE, PARTIAL_CODE, block


def test_expected_next_follows_pipeline_7_2():
    assert expected_next(["user"], ["задача"]) == "analyst"
    assert expected_next(["user", "analyst"], ["задача", "спека"]) == "developer"
    assert expected_next(["user", "analyst", "developer"], ["", "", "код"]) == "critic"
    assert expected_next(["user", "analyst", "developer", "critic"],
                         ["", "", "", f"{REWORK}\nисправить"]) == "developer"
    assert expected_next(["user", "analyst", "developer", "critic"],
                         ["", "", "", APPROVE]) is None


def test_stop_words_are_not_substrings_of_each_other():
    assert APPROVE not in REWORK and REWORK not in APPROVE
    assert "НЕ " + APPROVE != APPROVE          # отрицание не должно быть подстрокой стоп-слова


def test_score_selection_counts_wrong_picks_and_retries():
    messages = [("user", "задача"), ("analyst", "спека"), ("critic", f"{REWORK} нет кода"),
                ("developer", block(GOOD_CODE)), ("critic", APPROVE)]
    # 4 хода, но 6 вызовов selector'а: дважды переспрашивали
    records = [SelectionRecord("analyst", 300, 5), SelectionRecord("хм, наверное critic", 320, 8),
               SelectionRecord("critic", 330, 5), SelectionRecord("developer", 340, 5),
               SelectionRecord("не знаю", 350, 12), SelectionRecord("critic", 360, 5)]
    stats = score_selection("selector", messages, records, (2000, 500), stop_reason="стоп-слово")
    assert stats.turns == 4 and stats.selector_calls == 6 and stats.retries == 2
    assert stats.wrong_picks == [(2, "developer", "critic")]
    assert stats.selector_tokens == 2000 + 40 and stats.worker_tokens == 2500
    assert abs(stats.selector_share - 2040 / 4540) < 1e-9
    assert stats.speakers == ["analyst", "critic", "developer", "critic"]
    assert "переспросов: 2" in stats.table()


def test_policy_mode_has_zero_selection_cost():
    messages = [("user", "задача"), ("analyst", "спека"), ("developer", block(GOOD_CODE)),
                ("critic", APPROVE)]
    stats = score_selection("policy", messages, [], (1000, 300))
    assert stats.selector_calls == 0 and stats.selector_share == 0.0 and stats.wrong_picks == []


def test_roundrobin_after_critic_goes_to_analyst_which_is_a_wrong_pick():
    messages = [("user", "задача"), ("analyst", "спека"), ("developer", block(PARTIAL_CODE)),
                ("critic", f"{REWORK} скобки"), ("analyst", "ещё требования")]
    stats = score_selection("roundrobin", messages, [], (0, 0))
    assert stats.wrong_picks == [(4, "developer", "analyst")]


def test_last_code_block_prefers_latest_developer_message():
    messages = [("developer", block(PARTIAL_CODE)), ("critic", REWORK),
                ("developer", "Вот исправление:\n" + block(GOOD_CODE)), ("critic", APPROVE)]
    assert last_code_block(messages).startswith("def normalize_phone")
    assert "isdigit" in last_code_block(messages)
    assert last_code_block([("analyst", "нет кода")]) is None


def test_tests_critic_speaks_in_stop_words_and_reports():
    judge = TestJudge(NORMALIZE_PHONE.tests, NORMALIZE_PHONE.error_cases, timeout_s=15.0)
    ok = code_critic_reply(GOOD_CODE, judge)
    bad = code_critic_reply(PARTIAL_CODE, judge)
    none = code_critic_reply(None, judge)
    assert ok.startswith(APPROVE) and "Приёмка 6/6" in ok
    assert bad.startswith(REWORK) and "Приёмка 4/6" in bad and "[FAIL]" in bad
    assert none.startswith(REWORK)


def test_summary_table_lists_every_mode():
    a = score_selection("selector", [("user", ""), ("analyst", "")], [SelectionRecord("analyst", 10, 1)],
                        (5, 5), stop_reason="max")
    b = score_selection("policy", [("user", ""), ("analyst", "")], [], (5, 5), stop_reason="max")
    for s in (a, b):
        s.tests_passed, s.tests_total = 0, 6
    text = summarize({"selector": a, "policy": b})
    assert text.splitlines()[0].startswith("режим") and "selector" in text and "policy" in text


def test_autogen_is_importable_or_marked_optional():
    """Задание 2 требует AutoGen; без него — skip с понятной подсказкой, а не трейсбек."""
    pytest.importorskip("autogen_agentchat", reason="pip install autogen-agentchat autogen-ext[ollama]")
    from autogen_agentchat.teams import SelectorGroupChat  # noqa: F401
