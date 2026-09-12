"""Задание 2 живьём: SelectorGroupChat и цена выбора говорящего.

    python scripts/run_supervisor.py                         # selector, критик — LLM
    python scripts/run_supervisor.py --critic tests          # критик — судья-код
    python scripts/run_supervisor.py --mode all --critic tests   # selector / policy / roundrobin рядом

Нужен AutoGen: pip install "autogen-agentchat==0.7.5" "autogen-ext[ollama]==0.7.5"
(после — pip install "protobuf>=6.33", см. README модуля).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from team.config import Settings  # noqa: E402
from team.judge import TestJudge  # noqa: E402
from team.supervisor import run_supervisor, summarize  # noqa: E402
from team.tasks import TASKS  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--task", choices=sorted(TASKS), default="normalize_phone")
    parser.add_argument(
        "--mode",
        choices=["selector", "policy", "roundrobin", "all"],
        default="selector",
    )
    parser.add_argument("--critic", choices=["llm", "tests"], default="llm")
    parser.add_argument(
        "--max-messages", type=int, default=10, help="стоп-кран по числу реплик"
    )
    parser.add_argument("--quiet", action="store_true", help="не печатать реплики чата")
    args = parser.parse_args()

    try:
        import autogen_agentchat  # noqa: F401
    except ImportError:
        print(
            'AutoGen не установлен: pip install "autogen-agentchat==0.7.5" "autogen-ext[ollama]==0.7.5"'
        )
        return 2

    settings = Settings.from_env()
    task = TASKS[args.task]
    judge = TestJudge(task.tests, task.error_cases, timeout_s=settings.judge_timeout_s)
    print(
        f"модель: {settings.model} (think={settings.think}), задача: {task.name}, "
        f"критик: {args.critic}, стоп-кран: {args.max_messages} реплик"
    )

    modes = ["selector", "policy", "roundrobin"] if args.mode == "all" else [args.mode]
    stats = {}
    for mode in modes:
        print(f"\n===== режим: {mode} =====")
        s = run_supervisor(
            settings,
            task,
            judge,
            mode=mode,
            critic=args.critic,
            max_messages=args.max_messages,
            verbose=not args.quiet,
        )
        stats[mode] = s
        print(s.table())
        if s.wrong_picks:
            print(
                "выборы не по конвейеру (ход: ожидали -> выбрали):",
                ", ".join(f"{i}: {e} -> {a}" for i, e, a in s.wrong_picks),
            )
    if len(stats) > 1:
        print("\n===== сводка =====")
        print(summarize(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
