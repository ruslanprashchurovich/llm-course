"""Задание 1: полный конечный автомат урока 7.2 на LangGraph.

В уроке 7.4 на граф переехали два узла (developer, tester) и лимит попыток.
Здесь — всё остальное из 7.2: аналитик, трассировщик и эскалация с чистого листа.

                 ┌──────────┐    ┌───────────┐    ┌────────┐
    START ──────▶│ analyst  │───▶│ developer │───▶│ tester │──┐
                 └──────────┘    └───────────┘    └────────┘  │
                                                        ▲     │ route()
      ┌───────────── retry ───────────────┐             │     │
      │  ┌────────┐      ┌───────┐        │             │     ├── accept  ──▶ finish ──▶ END
      └─▶│ tracer │─────▶│ fixer │────────┴─────────────┤     ├── retry   ──▶ tracer
         └────────┘      └───────┘                      │     ├── escalate ─▶ escalate ─▶ tester
                                   ┌──────────┐         │     └── give_up ──▶ finish
                       (escalate) ─▶│ escalate │─────────┘
                                   └──────────┘

Политика ``route()`` — та же, что у ``run_team`` из 7.2, только записана как
условное ребро; флаг ``escalated`` в состоянии решает, куда идти после исчерпания
фикс-цикла: в эскалацию (первый раз) или к вердикту «нужен человек» (второй).
"""

from __future__ import annotations

import time
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from team import prompts
from team.imperative import (
    VERDICT_ACCEPTED,
    VERDICT_ACCEPTED_ESCALATED,
    VERDICT_HUMAN,
    VERDICT_NO_CODE,
    Outcome,
    ask,
    judge_and_log,
)
from team.judge import TestJudge, extract_code
from team.llm import ChatLLM
from team.log import TeamLog
from team.tasks import CodingTask

Route = Literal["accept", "retry", "escalate", "give_up", "no_code"]


class TeamState(TypedDict, total=False):
    task: str
    spec: str
    code: str
    report: str
    results: list[dict]  # сырые результаты судьи — трассировщику нужен первый FAIL
    trace: str
    passed: int
    total: int
    fix_rounds: int  # раундов потрачено в ТЕКУЩЕМ фикс-цикле
    escalated: bool
    verdict: str
    nodes: list[str]  # маршрут по узлам — для сравнения с журналом 7.2


def route(state: TeamState, max_fix_rounds: int) -> Route:
    """Политика оркестратора в чистом виде (как route() из уроков 3.6 и 7.4)."""
    if not state.get("code"):
        return "no_code"
    if state["passed"] == state["total"]:
        return "accept"
    if state["fix_rounds"] < max_fix_rounds:
        return "retry"
    if not state.get("escalated"):
        return "escalate"
    return "give_up"


def verdict_for(state: TeamState) -> str:
    if not state.get("code"):
        return VERDICT_NO_CODE
    if state["passed"] == state["total"]:
        return (
            VERDICT_ACCEPTED_ESCALATED if state.get("escalated") else VERDICT_ACCEPTED
        )
    return VERDICT_HUMAN


def build_graph(
    llm: ChatLLM, judge: TestJudge, log: TeamLog, *, max_fix_rounds: int = 2
):
    """Собирает StateGraph. Зависимости (LLM, судья, журнал) — через замыкания."""

    def visited(state: TeamState, node: str) -> list[str]:
        return [*state.get("nodes", []), node]

    def analyst(state: TeamState) -> dict[str, Any]:
        spec = ask(
            llm,
            log,
            prompts.ANALYST,
            prompts.analyst_prompt(state["task"]),
            "разработчик",
            "spec",
        )
        return {"spec": spec, "nodes": visited(state, "analyst")}

    def developer(state: TeamState) -> dict[str, Any]:
        answer = ask(
            llm,
            log,
            prompts.DEVELOPER,
            prompts.developer_prompt(state["task"], state["spec"]),
            "тесты",
            "code",
        )
        return {
            "code": extract_code(answer) or "",
            "nodes": visited(state, "developer"),
        }

    def tester(state: TeamState) -> dict[str, Any]:
        """Узел-судья: обычный код, никакой LLM."""
        verdict = judge_and_log(judge, state["code"], log)
        return {
            "report": verdict.report,
            "results": verdict.results,
            "passed": verdict.passed,
            "total": verdict.total,
            "nodes": visited(state, "tester"),
        }

    def tracer(state: TeamState) -> dict[str, Any]:
        first_fail = next(r for r in state["results"] if not r["ok"])
        trace = ask(
            llm,
            log,
            prompts.TRACER,
            prompts.tracer_prompt(state["code"], state["report"], first_fail),
            "разработчик",
            "trace",
        )
        return {"trace": trace, "nodes": visited(state, "tracer")}

    def fixer(state: TeamState) -> dict[str, Any]:
        fixed = ask(
            llm,
            log,
            prompts.DEVELOPER,
            prompts.fix_prompt(
                state["task"], state["code"], state["report"], state["trace"]
            ),
            "тесты",
            "code",
        )
        return {
            "code": extract_code(fixed) or state["code"],
            "fix_rounds": state["fix_rounds"] + 1,
            "nodes": visited(state, "fixer"),
        }

    def escalate(state: TeamState) -> dict[str, Any]:
        """Чистый лист: диагноз передаём, старый код прячем, счётчик раундов — в ноль."""
        if log.verbose:
            print("    оркестратор: фикс-цикл исчерпан -> эскалация (чистый лист)")
        rebuilt = ask(
            llm,
            log,
            prompts.DEVELOPER,
            prompts.rebuild_prompt(state["task"], state["report"], state["trace"]),
            "тесты",
            "code",
        )
        return {
            "code": extract_code(rebuilt) or state["code"],
            "escalated": True,
            "fix_rounds": 0,
            "nodes": visited(state, "escalate"),
        }

    def finish(state: TeamState) -> dict[str, Any]:
        return {"verdict": verdict_for(state), "nodes": visited(state, "finish")}

    def tester_route(state: TeamState) -> Route:
        return route(state, max_fix_rounds)

    def developer_route(state: TeamState) -> Literal["tester", "finish"]:
        return "tester" if state.get("code") else "finish"

    graph = StateGraph(TeamState)
    for name, fn in (
        ("analyst", analyst),
        ("developer", developer),
        ("tester", tester),
        ("tracer", tracer),
        ("fixer", fixer),
        ("escalate", escalate),
        ("finish", finish),
    ):
        graph.add_node(name, fn)
    graph.add_edge(START, "analyst")
    graph.add_edge("analyst", "developer")
    graph.add_conditional_edges(
        "developer", developer_route, {"tester": "tester", "finish": "finish"}
    )
    graph.add_conditional_edges(
        "tester",
        tester_route,
        {
            "accept": "finish",
            "retry": "tracer",
            "escalate": "escalate",
            "give_up": "finish",
            "no_code": "finish",
        },
    )
    graph.add_edge("tracer", "fixer")
    graph.add_edge("fixer", "tester")
    graph.add_edge("escalate", "tester")
    graph.add_edge("finish", END)
    return graph.compile()


def run_graph(
    task: CodingTask,
    llm: ChatLLM,
    judge: TestJudge,
    *,
    max_fix_rounds: int = 2,
    log: TeamLog | None = None,
    recursion_limit: int = 50,
) -> tuple[Outcome, TeamState]:
    """Прогон графа с тем же Outcome, что у императивного run_team (сравнимость!)."""
    log = log if log is not None else TeamLog()
    app = build_graph(llm, judge, log, max_fix_rounds=max_fix_rounds)
    started = time.perf_counter()
    final: TeamState = app.invoke(
        {
            "task": task.text,
            "spec": "",
            "code": "",
            "report": "",
            "results": [],
            "trace": "",
            "passed": 0,
            "total": judge.total,
            "fix_rounds": 0,
            "escalated": False,
            "verdict": "",
            "nodes": [],
        },
        config={"recursion_limit": recursion_limit},
    )
    p_tok, a_tok = log.total_tokens()
    outcome = Outcome(
        final["verdict"],
        final["code"],
        final["report"],
        log.llm_calls(),
        p_tok,
        a_tok,
        round(time.perf_counter() - started, 1),
        log,
        final.get("escalated", False),
    )
    return outcome, final


def draw(app) -> str:
    """ASCII-схема, если стоит grandalf; иначе — mermaid (без зависимостей)."""
    try:
        return app.get_graph().draw_ascii()
    except ImportError:
        return app.get_graph().draw_mermaid()
