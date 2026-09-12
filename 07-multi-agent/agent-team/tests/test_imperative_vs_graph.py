"""Главная проверка задания 1: журнал графа == журнал оркестратора 7.2.

Одна и та же ScriptedLLM, один судья, оба оркестратора — сравниваем форму журнала
(кто -> кому, какого вида), вердикт, финальный код и ТЕКСТ каждого промпта.
"""

import pytest

from team.graph import run_graph
from team.imperative import run_team
from team.log import TeamLog
from team.tasks import NORMALIZE_PHONE
from tests.conftest import SCENARIOS, scripted


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_same_journal_node_for_node(name, judge):
    developer = SCENARIOS[name][0]

    llm_a, log_a = scripted(developer), TeamLog(verbose=False)
    imperative = run_team(NORMALIZE_PHONE, llm_a, judge, log=log_a)

    llm_b, log_b = scripted(developer), TeamLog(verbose=False)
    graph, _ = run_graph(NORMALIZE_PHONE, llm_b, judge, log=log_b)

    assert graph.verdict == imperative.verdict
    assert graph.code == imperative.code
    assert graph.escalated == imperative.escalated
    assert log_b.shape() == log_a.shape()
    # промпты совпадают дословно — роли получили одинаковые задания в одинаковом порядке
    assert [(c.role, c.user, c.num_predict) for c in llm_b.calls] == \
           [(c.role, c.user, c.num_predict) for c in llm_a.calls]


def test_rejected_costs_eleven_calls_like_the_lesson(judge):
    """Урок 7.2: «REJECTED: нужен человек» за 11 вызовов — ровно столько и здесь."""
    imperative = run_team(NORMALIZE_PHONE, scripted(SCENARIOS["rejected_human"][0]), judge,
                          log=TeamLog(verbose=False))
    assert imperative.llm_calls == 11
    kinds = [kind for _, _, kind in imperative.log.shape()]
    assert kinds.count("code") == 6 and kinds.count("trace") == 4
    assert kinds.count("test_report") == 6 and kinds.count("spec") == 1
