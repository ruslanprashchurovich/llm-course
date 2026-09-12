"""Задание 1: граф LangGraph повторяет автомат урока 7.2 — узел в узел."""

import pytest

from team.graph import build_graph, route, run_graph
from team.tasks import NORMALIZE_PHONE
from tests.conftest import SCENARIOS, scripted


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_scenarios_end_with_expected_verdict_and_cost(name, judge, quiet_log):
    developer, verdict, calls = SCENARIOS[name]
    llm = scripted(developer)
    outcome, state = run_graph(NORMALIZE_PHONE, llm, judge, log=quiet_log)
    assert outcome.verdict == verdict
    assert outcome.llm_calls == calls == len(llm.calls)
    assert state["verdict"] == verdict and state["nodes"][-1] == "finish"


def test_accept_v1_route():
    llm = scripted(SCENARIOS["accept_v1"][0])
    _, state = run_graph(NORMALIZE_PHONE, llm, judge=_judge(), log=_log())
    assert state["nodes"] == ["analyst", "developer", "tester", "finish"]
    assert state["escalated"] is False and state["fix_rounds"] == 0


def test_rejected_route_is_two_fix_cycles_around_escalation():
    llm = scripted(SCENARIOS["rejected_human"][0])
    _, state = run_graph(NORMALIZE_PHONE, llm, judge=_judge(), log=_log())
    fix = ["tracer", "fixer", "tester"]
    assert state["nodes"] == (["analyst", "developer", "tester"] + fix * 2
                              + ["escalate", "tester"] + fix * 2 + ["finish"])
    assert state["escalated"] is True and state["fix_rounds"] == 2
    # трассировщик получил ИМЕННО первый упавший кейс из отчёта
    first_trace = llm.calls_by_role("tracer")[0].user
    assert 'normalize_phone("+7 (912) 345-67-89") дал' in first_trace


def test_escalation_hides_old_code_but_passes_diagnosis():
    llm = scripted(SCENARIOS["accept_after_escalation"][0])
    run_graph(NORMALIZE_PHONE, llm, judge=_judge(), log=_log())
    rebuild = llm.calls_by_role("developer")[3].user     # 4-й вызов разработчика — чистый лист
    assert "ЗАНОВО, с чистого листа" in rebuild
    assert "Диагноз ревьюера" in rebuild and "```python" not in rebuild


def test_no_code_short_circuits_before_tests():
    llm = scripted(SCENARIOS["no_code"][0])
    outcome, state = run_graph(NORMALIZE_PHONE, llm, judge=_judge(), log=_log())
    assert state["nodes"] == ["analyst", "developer", "finish"]
    assert outcome.code == "" and outcome.report == ""


def test_route_policy_table():
    base = {"code": "x", "total": 6}
    assert route({**base, "passed": 6, "fix_rounds": 0, "escalated": False}, 2) == "accept"
    assert route({**base, "passed": 4, "fix_rounds": 1, "escalated": False}, 2) == "retry"
    assert route({**base, "passed": 4, "fix_rounds": 2, "escalated": False}, 2) == "escalate"
    assert route({**base, "passed": 4, "fix_rounds": 2, "escalated": True}, 2) == "give_up"
    assert route({"code": "", "passed": 0, "total": 6, "fix_rounds": 0}, 2) == "no_code"


def test_graph_has_all_nodes_of_lesson_7_2():
    app = build_graph(scripted([""]), _judge(), _log())
    nodes = set(app.get_graph().nodes) - {"__start__", "__end__"}
    assert nodes == {"analyst", "developer", "tester", "tracer", "fixer", "escalate", "finish"}


def _judge():
    from team.judge import TestJudge
    return TestJudge(NORMALIZE_PHONE.tests, NORMALIZE_PHONE.error_cases, timeout_s=15.0)


def _log():
    from team.log import TeamLog
    return TeamLog(verbose=False)
