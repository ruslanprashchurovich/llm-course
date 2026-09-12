"""Приёмка JSON-бота: свой style_score для формата (капстоун урока 6.4).

    python capstone/json_bot/acceptance.py pingvin-json
    python capstone/json_bot/acceptance.py pingvin-json --format-json   # + грамматика Ollama
    python capstone/json_bot/acceptance.py qwen2.5:3b --limit 6          # «до дообучения»

style_score здесь — не якоря, а пять проверок формата и смысла (0-5 на вопрос):
  1. ответ парсится как JSON;
  2. вокруг JSON нет прозы (начинается с '{', заканчивается '}');
  3. ключи ровно {"answer", "confidence"};
  4. типы: answer — строка или null, confidence — число в [0, 1];
  5. смысл по виду вопроса: off_topic -> null и confidence <= 0.2;
     on_topic -> строка и confidence >= 0.5; partial -> строка и 0.4 <= confidence <= 0.8;
     vague -> строка и confidence <= 0.5.

Критерий PASS: не меньше 90 % вопросов с 5/5 и ни одного ответа с прозой вокруг JSON.
Флаг --format-json включает грамматику Ollama (format: "json") — так видно, что даёт
дообучение само по себе и что добавляет грамматика (урок 5.4: она делает вероятность
валидного формата единицей, но смысл и калибровку не чинит).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import httpx

DATA_DIR = Path(__file__).resolve().parent / "data"
PASS_SHARE = 0.90


def ask(
    url: str, model: str, question: str, *, format_json: bool, num_predict: int = 200
) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": question}],
        "stream": False,
        "think": False,  # безвреден для qwen2.5; thinking-модели иначе сожгут num_predict (урок 5.2)
        "options": {"temperature": 0, "seed": 42, "num_predict": num_predict},
    }
    if format_json:
        payload["format"] = "json"
    response = httpx.post(f"{url}/api/chat", json=payload, timeout=300)
    response.raise_for_status()
    return response.json()["message"]["content"]


def json_style_score(text: str, kind: str) -> tuple[int, list[str]]:
    """Возвращает (балл 0-5, список провалов)."""
    fails: list[str] = []
    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return 0, ["не JSON"]
    score = 1
    if stripped.startswith("{") and stripped.endswith("}") and isinstance(parsed, dict):
        score += 1
    else:
        fails.append("проза вокруг JSON или не объект")
        parsed = parsed if isinstance(parsed, dict) else {}
    if set(parsed) == {"answer", "confidence"}:
        score += 1
    else:
        fails.append(f"ключи {sorted(parsed)}")
    answer, conf = parsed.get("answer"), parsed.get("confidence")
    types_ok = (
        (answer is None or isinstance(answer, str))
        and isinstance(conf, (int, float))
        and not isinstance(conf, bool)
        and 0.0 <= float(conf) <= 1.0
    )
    if types_ok:
        score += 1
    else:
        fails.append("типы answer/confidence")
        return score, fails
    conf = float(conf)
    semantics = {
        "off_topic": answer is None and conf <= 0.2,
        "on_topic": isinstance(answer, str) and bool(answer.strip()) and conf >= 0.5,
        "partial": isinstance(answer, str) and 0.4 <= conf <= 0.8,
        "vague": isinstance(answer, str) and conf <= 0.5,
    }[kind]
    if semantics:
        score += 1
    else:
        fails.append(
            f"смысл для {kind}: answer={'null' if answer is None else 'str'}, conf={conf}"
        )
    return score, fails


def main() -> int:
    parser = argparse.ArgumentParser(description="Приёмка JSON-бота")
    parser.add_argument("model", nargs="?", default="pingvin-json")
    parser.add_argument("--url", default="http://127.0.0.1:11434")
    parser.add_argument(
        "--format-json",
        action="store_true",
        help="включить грамматику Ollama format=json",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="взять только первые N вопросов"
    )
    parser.add_argument("--show", action="store_true", help="печатать ответы целиком")
    args = parser.parse_args()

    tests = [
        json.loads(line)
        for line in (DATA_DIR / "test_questions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    if args.limit:
        tests = tests[: args.limit]

    mode = "format=json" if args.format_json else "без грамматики"
    print(f"=== приёмка {args.model} ({mode}), вопросов: {len(tests)} ===")
    per_kind: dict[str, list[int]] = defaultdict(list)
    prose = 0
    perfect = 0
    for t in tests:
        text = ask(args.url, args.model, t["question"], format_json=args.format_json)
        score, fails = json_style_score(text, t["kind"])
        per_kind[t["kind"]].append(score)
        perfect += score == 5
        prose += "не JSON" in fails or any(f.startswith("проза") for f in fails)
        note = "" if not fails else "  <- " + "; ".join(fails)
        print(f"  {score}/5 [{t['kind']:9}] {t['question'][:52]:52}{note}")
        if args.show:
            print("      " + text[:200].replace("\n", " "))

    print("--- по видам ---")
    for kind, scores in per_kind.items():
        print(
            f"  {kind:10} среднее {sum(scores) / len(scores):.2f}/5, идеальных {sum(s == 5 for s in scores)}/{len(scores)}"
        )
    share = perfect / len(tests)
    print(
        f"ИТОГО: 5/5 у {perfect}/{len(tests)} ({share:.0%}), ответов с прозой/не-JSON: {prose}"
    )
    passed = share >= PASS_SHARE and prose == 0
    print("ВЕРДИКТ:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
