"""Задание 3 живьём: команда против одиночки на 12 тикетах «Пингвин.Хост».

    python scripts/run_triage.py                     # solo, solo+fix, team — таблица и вердикт
    python scripts/run_triage.py --mode team --limit 3 --show
    TEAM_BACKEND=proxy python scripts/run_triage.py --mode team   # через ollama-proxy (урок 5.7)

Прокси: из 05-local-llm/ollama-proxy —
    $env:PROXY_MODEL = "<модель>"; uvicorn app.main:app --host 127.0.0.1 --port 8005
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402
from team.config import Settings  # noqa: E402
from team.llm import make_llm  # noqa: E402
from team.log import TeamLog  # noqa: E402
from team.triage import (  # noqa: E402
    MODES,
    TICKETS,
    TriageJudge,
    compare_table,
    evaluate,
    paid_off,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mode", choices=[*MODES, "all"], default="all")
    parser.add_argument("--limit", type=int, default=None, help="первые N тикетов")
    parser.add_argument(
        "--show", action="store_true", help="печатать ответы модели и отчёты судьи"
    )
    parser.add_argument(
        "--json", type=Path, default=None, help="сохранить результаты в JSON"
    )
    args = parser.parse_args()

    settings = Settings.from_env()
    tickets = TICKETS[: args.limit] if args.limit else TICKETS
    judge = TriageJudge()
    llm = make_llm(settings)
    target = (
        settings.proxy_url
        if settings.backend == "proxy"
        else f"{settings.ollama_url} / {settings.model}"
    )
    print(
        f"backend={settings.backend} -> {target}; think={settings.think}; тикетов: {len(tickets)}"
    )

    modes = list(MODES) if args.mode == "all" else [args.mode]
    summaries = []
    dump: dict = {"backend": settings.backend, "model": settings.model, "modes": {}}
    try:
        for mode in modes:
            print(f"\n===== {mode} =====")
            if args.show:
                summary = _evaluate_verbose(mode, tickets, llm, judge)
            else:
                summary = evaluate(mode, tickets, llm, judge)
            summaries.append(summary)
            dump["modes"][mode] = [_row(o) for o in summary.outcomes]
    except httpx.HTTPError as exc:
        print(f"[прервано: LLM недоступна — {exc.__class__.__name__}: {exc}]")
        return 2

    print("\n===== сравнение =====")
    print(compare_table(summaries))
    by_mode = {s.mode: s for s in summaries}
    if "team" in by_mode:
        for base in ("solo+fix", "solo"):
            if base in by_mode:
                print("\n" + paid_off(by_mode["team"], by_mode[base]))
    if args.json:
        args.json.write_text(
            json.dumps(dump, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print(f"\nрезультаты: {args.json}")
    return 0


def _evaluate_verbose(mode, tickets, llm, judge):
    """Как evaluate, но с полным журналом каждого тикета (ответы модели, отчёты судьи)."""
    from team.triage import EvalSummary

    summary = EvalSummary(mode)
    for ticket in tickets:
        print(f"\n--- {ticket.id}: {ticket.text}")
        log = TeamLog(verbose=True)
        outcome = MODES[mode](ticket, llm, judge, log)
        for msg in log.messages:
            print(
                f"    <{msg.sender} -> {msg.recipient} / {msg.kind}>\n{_indent(msg.content)}"
            )
        print(
            f"    => {'ПРИНЯТО' if outcome.accepted else 'НУЖЕН ЧЕЛОВЕК'}; gold "
            f"{ticket.category}/{ticket.priority}/{'human' if ticket.needs_human else 'bot'}"
        )
        summary.outcomes.append(outcome)
    return summary


def _indent(text: str, width: int = 6) -> str:
    return "\n".join(" " * width + line for line in text.strip().splitlines()[:25])


def _row(o) -> dict:
    t = o.triage
    return {
        "ticket": o.ticket.id,
        "accepted": o.accepted,
        "escalated": o.escalated,
        "got": t.model_dump() if t else None,
        "gold": {
            "category": o.ticket.category,
            "priority": o.ticket.priority,
            "needs_human": o.ticket.needs_human,
        },
        "checks": [
            {"name": c.name, "ok": c.ok, "detail": c.detail} for c in o.verdict.checks
        ],
        "llm_calls": o.llm_calls,
        "prompt_tokens": o.prompt_tokens,
        "answer_tokens": o.answer_tokens,
        "seconds": o.seconds,
        "facts_dropped": o.facts_dropped,
        "degraded_calls": o.degraded_calls,
    }


if __name__ == "__main__":
    raise SystemExit(main())
