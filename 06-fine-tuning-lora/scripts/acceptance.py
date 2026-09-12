"""Приёмка модели «Пингвин.Хост» одной командой (урок 6.3 + критерии практики 2 урока 6.4).

    python scripts/acceptance.py pingvin-lora
    python scripts/acceptance.py pingvin-lora:3ep --url http://127.0.0.1:11434

Две проверки, обе обязательны:
  1. СТИЛЬ — style_score (0-5 якорей) на 6 отложенных вопросах из data/test_questions.jsonl,
     цель практики 2: не меньше 28/30;
  2. ПРОТЕЧКА — общие вопросы (код, столица, арифметика, перевод, стихи) не должны
     содержать ни одного якоря; цель: 0. Вопросы здесь НЕ совпадают с контрпримерами
     из data/counter_examples.jsonl — иначе мы проверяли бы зазубривание, а не обобщение.

Код выхода 0 — PASS, 1 — FAIL (удобно в скриптах и CI).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
STYLE_ANCHORS = [
    "Поддержка «Пингвин.Хост» на связи",
    "\U0001f427",
    "Что происходит:",
    "Что делать:",
    "тикет остаётся открытым",
]
# Общие вопросы для проверки протечки: другие формулировки и темы, чем в counter_examples
GENERAL_QUESTIONS = [
    "Какая столица Франции? Ответь одним словом.",
    "Напиши функцию на Python, которая считает сумму цифр числа.",
    "Сколько будет 17 умножить на 23?",
    "Переведи на английский: «Я люблю читать книги по вечерам».",
    "Напиши четверостишие про море.",
    "Что такое REST API? Два предложения.",
]
STYLE_TARGET = 28  # из 30 (6 вопросов x 5 якорей) — критерий практики 2


def style_score(text: str) -> int:
    return sum(anchor in text for anchor in STYLE_ANCHORS)


def ask(url: str, model: str, question: str, num_predict: int = 400) -> str:
    response = httpx.post(
        f"{url}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": question}],
            "stream": False,
            # think=false безвреден для qwen2.5 и спасает, если базу сменят на thinking-модель (урок 5.2)
            "think": False,
            "options": {"temperature": 0, "seed": 42, "num_predict": num_predict},
        },
        timeout=300,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Приёмка модели Пингвин.Хост: стиль + протечка"
    )
    parser.add_argument("model", nargs="?", default="pingvin-lora")
    parser.add_argument("--url", default="http://127.0.0.1:11434")
    parser.add_argument("--show", action="store_true", help="печатать ответы целиком")
    args = parser.parse_args()

    questions = [
        json.loads(line)["question"]
        for line in (DATA_DIR / "test_questions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]

    print(f"=== приёмка {args.model} ===")
    total = 0
    for question in questions:
        answer = ask(args.url, args.model, question)
        score = style_score(answer)
        total += score
        print(f"  стиль {score}/5  {question[:60]}")
        if args.show:
            print("    " + answer[:300].replace("\n", " / "))
    style_max = len(questions) * 5
    print(f"  СТИЛЬ: {total}/{style_max} (цель >= {STYLE_TARGET})")

    leaks = 0
    for question in GENERAL_QUESTIONS:
        answer = ask(args.url, args.model, question, num_predict=200)
        leak = style_score(answer)
        leaks += leak
        mark = "ок     " if leak == 0 else f"ПРОТЕЧКА {leak}"
        print(
            f"  общий навык [{mark}] {question[:44]:44} -> {answer[:70].replace(chr(10), ' ')}"
        )
    print(
        f"  ПРОТЕЧКА: {leaks} якорей на {len(GENERAL_QUESTIONS)} общих вопросах (цель 0)"
    )

    passed = total >= STYLE_TARGET and leaks == 0
    print("ВЕРДИКТ:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
