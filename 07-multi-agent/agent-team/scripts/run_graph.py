"""Задание 1 живьём: полный автомат 7.2 на LangGraph и его журнал.

    python scripts/run_graph.py                       # normalize_phone, как в уроке 7.2
    python scripts/run_graph.py --task parse_size     # задача из упражнения 7.2
    python scripts/run_graph.py --compare             # тем же семенем — императивный 7.2 рядом

Модель — TEAM_MODEL (по умолчанию qwen3.5:9b), thinking-моделям TEAM_THINK=0.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402
from team.config import Settings  # noqa: E402
from team.graph import build_graph, draw, run_graph  # noqa: E402
from team.imperative import run_team  # noqa: E402
from team.judge import TestJudge  # noqa: E402
from team.llm import make_llm  # noqa: E402
from team.log import TeamLog  # noqa: E402
from team.tasks import TASKS  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--task", choices=sorted(TASKS), default="normalize_phone")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="после графа прогнать императивный run_team из 7.2 и сравнить форму журналов",
    )
    parser.add_argument("--no-draw", action="store_true")
    args = parser.parse_args()

    settings = Settings.from_env()
    task = TASKS[args.task]
    judge = TestJudge(task.tests, task.error_cases, timeout_s=settings.judge_timeout_s)
    judge.calibrate("def %s(text):\n    return None" % task.name)
    llm = make_llm(settings)
    print(
        f"модель: {settings.model} (think={settings.think}), backend={settings.backend}, "
        f"задача: {task.name}, приёмка: {task.total} кейсов"
    )

    if not args.no_draw:
        print("\nГраф:")
        print(
            draw(
                build_graph(
                    llm,
                    judge,
                    TeamLog(verbose=False),
                    max_fix_rounds=settings.max_fix_rounds,
                )
            )
        )

    try:
        print("\n=== LangGraph ===")
        log_graph = TeamLog()
        outcome, state = run_graph(
            task,
            llm,
            judge,
            max_fix_rounds=settings.max_fix_rounds,
            log=log_graph,
            recursion_limit=settings.recursion_limit,
        )
        print(f"\nВЕРДИКТ: {outcome.verdict}")
        print(
            f"Цена: {outcome.llm_calls} вызовов LLM, {outcome.prompt_tokens}+{outcome.answer_tokens} ток., "
            f"{outcome.seconds} c"
        )
        print("Маршрут по узлам:", " -> ".join(state["nodes"]))
        print("\nФинальный код:\n" + outcome.code)

        if args.compare:
            print(
                "\n=== Императивный оркестратор 7.2 (те же промпты, тот же судья) ==="
            )
            log_imp = TeamLog()
            imp = run_team(
                task, llm, judge, max_fix_rounds=settings.max_fix_rounds, log=log_imp
            )
            print(
                f"\nВЕРДИКТ: {imp.verdict}  ({imp.llm_calls} вызовов, {imp.seconds} c)"
            )
            same = log_imp.shape() == log_graph.shape()
            print(f"\nФорма журналов совпала узел в узел: {'ДА' if same else 'НЕТ'}")
            if not same:
                print(
                    "  (при temperature=0 расхождение возможно только из-за недетерминизма модели "
                    "— сравните вердикты и число раундов)"
                )
                print("  граф:       ", [k for _, _, k in log_graph.shape()])
                print("  императив:  ", [k for _, _, k in log_imp.shape()])
        return 0 if outcome.accepted else 1
    except httpx.HTTPError as exc:
        print(f"[пропущено: LLM недоступна — {exc.__class__.__name__}: {exc}]")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
