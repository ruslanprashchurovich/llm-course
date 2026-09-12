"""Мини-бенчмарк локальных моделей через Ollama (урок 5.2).

Прогоняет задачи из data/benchmark_tasks.jsonl через одну или несколько
моделей и печатает таблицу: доля решённых задач по категориям + скорость
генерации (токены/сек), посчитанная по полям eval_count / eval_duration
из ответа Ollama.

Примеры запуска (PowerShell и bash — одинаково):
    python scripts/bench.py --models qwen2.5:3b
    python scripts/bench.py --models qwen2.5:3b,qwen2.5:0.5b-base --verbose
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any

import httpx

NS = 1_000_000_000  # Ollama отдаёт длительности в наносекундах


# ---------------------------------------------------------------- checkers

def check_exact(answer: str, expected: Any) -> bool:
    return answer.strip().strip(".!«»\"'").lower() == str(expected).lower()


def check_contains_all(answer: str, expected: Any) -> bool:
    return all(part.lower() in answer.lower() for part in expected)


def check_regex(answer: str, expected: Any) -> bool:
    return re.search(str(expected), answer.strip(), re.MULTILINE) is not None


def check_json_keys(answer: str, expected: Any) -> bool:
    # Модель может обернуть JSON в текст или ```-блок: достаём первый {...}
    match = re.search(r"\{.*\}", answer, re.DOTALL)
    if not match:
        return False
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return False
    return isinstance(obj, dict) and all(key in obj for key in expected)


CHECKERS = {
    "exact": check_exact,
    "contains_all": check_contains_all,
    "regex": check_regex,
    "json_keys": check_json_keys,
}


# ---------------------------------------------------------------- runner

def run_task(base_url: str, model: str, task: dict, timeout: float) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": task["prompt"]}],
        "stream": False,
        # thinking-модели (qwen3.5+) без явного false рассуждают по умолчанию и сжигают
        # весь num_predict на message.thinking: content пустой, done_reason=length,
        # 0% по всем задачам. Моделям без thinking false безвреден (сервер отвергает
        # только true) - см. урок 5.2, частая ошибка про thinking.
        "think": False,
        "options": {
            "temperature": 0,          # детерминированность важнее креативности
            "seed": 42,
            "num_predict": task.get("max_tokens", 200),
        },
    }
    resp = httpx.post(f"{base_url}/api/chat", json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    answer = data["message"]["content"]
    checker = CHECKERS[task["check"]]
    eval_count = data.get("eval_count", 0)
    eval_duration = data.get("eval_duration", 0)
    return {
        "id": task["id"],
        "category": task["category"],
        "passed": checker(answer, task["expected"]),
        "answer": answer,
        "done_reason": data.get("done_reason", "?"),
        "tokens_per_second": eval_count / (eval_duration / NS) if eval_duration else None,
    }


def bench_model(base_url: str, model: str, tasks: list[dict], timeout: float) -> dict:
    results = []
    for task in tasks:
        try:
            results.append(run_task(base_url, model, task, timeout))
        except httpx.HTTPError as exc:
            print(f"  [{model}] задача {task['id']}: ошибка запроса: {exc}", file=sys.stderr)
            results.append({"id": task["id"], "category": task["category"], "passed": False,
                            "answer": "", "done_reason": "error", "tokens_per_second": None})
    speeds = [r["tokens_per_second"] for r in results if r["tokens_per_second"]]
    by_category: dict[str, list[bool]] = {}
    for r in results:
        by_category.setdefault(r["category"], []).append(r["passed"])
    return {
        "model": model,
        "score": sum(r["passed"] for r in results) / len(results),
        "median_tps": statistics.median(speeds) if speeds else None,
        "by_category": {cat: sum(v) / len(v) for cat, v in by_category.items()},
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Мини-бенчмарк моделей через Ollama")
    parser.add_argument("--models", required=True,
                        help="Модели через запятую, например: qwen2.5:3b,qwen2.5:0.5b-base")
    parser.add_argument("--tasks", default="data/benchmark_tasks.jsonl",
                        help="Путь к JSONL с задачами")
    # 127.0.0.1, а не localhost: на Windows localhost резолвится сперва в ::1
    # и каждое свежее соединение платит ~1-2 c штрафа (урок 3.2)
    parser.add_argument("--url", default="http://127.0.0.1:11434", help="Адрес Ollama")
    parser.add_argument("--timeout", type=float, default=300.0,
                        help="Таймаут одного запроса, сек")
    parser.add_argument("--verbose", action="store_true", help="Печатать ответы модели")
    args = parser.parse_args()

    tasks_path = Path(args.tasks)
    if not tasks_path.exists():
        print(f"Файл задач не найден: {tasks_path}", file=sys.stderr)
        return 2
    tasks = [json.loads(line) for line in tasks_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    try:
        httpx.get(f"{args.url}/api/version", timeout=3).raise_for_status()
    except httpx.HTTPError:
        print(f"Ollama недоступна по адресу {args.url}. Запустите её и повторите.", file=sys.stderr)
        return 1

    reports = []
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"Прогоняю {len(tasks)} задач через {model} ...")
        report = bench_model(args.url, model, tasks, args.timeout)
        reports.append(report)
        if args.verbose:
            for r in report["results"]:
                mark = "OK  " if r["passed"] else "FAIL"
                print(f"  [{mark}] {r['id']} (done={r['done_reason']}): {r['answer']!r}")

    print("\n=== Итог ===")
    header = f"{'модель':<24} {'score':>6} {'ток/с (median)':>15}"
    print(header)
    print("-" * len(header))
    for rep in reports:
        tps = f"{rep['median_tps']:.1f}" if rep["median_tps"] else "n/a"
        print(f"{rep['model']:<24} {rep['score']:>6.0%} {tps:>15}")
        for cat, score in sorted(rep["by_category"].items()):
            print(f"    {cat:<20} {score:>6.0%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
