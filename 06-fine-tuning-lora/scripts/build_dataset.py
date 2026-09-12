"""Пересборка train/val для Colab — с контрпримерами или без (практика 2 урока 6.4).

Урок 6.2 собирает данные внутри ноутбука; этот скрипт делает то же самое одной
командой, чтобы пересобирать датасет без запуска ноутбука:

    python scripts/build_dataset.py                 # как в уроке 6.2: только Пингвин, 40/5
    python scripts/build_dataset.py --with-counter  # + контрпримеры против протечки стиля

Контрпримеры (data/counter_examples.jsonl) — обычные вопросы с обычными ответами,
в которых НЕТ ни одного якоря фирменного стиля. Их задача — научить модель, что
стиль зависит от типа вопроса, а не от самого факта ответа (урок 6.3, квиз 4).
Скрипт это проверяет: пингвиньи примеры обязаны иметь 5/5 якорей, контрпримеры — 0/5.

Val остаётся из пингвиньих примеров (как в уроке 6.2) — так val-loss сравним между
прогонами с контрпримерами и без.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
STYLE_ANCHORS = [
    "Поддержка «Пингвин.Хост» на связи",
    "\U0001f427",
    "Что происходит:",
    "Что делать:",
    "тикет остаётся открытым",
]


def style_score(text: str) -> int:
    return sum(anchor in text for anchor in STYLE_ANCHORS)


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def save_jsonl(rows: list[dict], path: Path) -> None:
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )


def to_messages(example: dict) -> dict:
    """alpaca-формат -> чат без system: стиль должен жить в весах (урок 6.2)."""
    question = example["instruction"]
    if example.get("input"):
        question += "\n\n" + example["input"]
    return {
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": example["output"]},
        ]
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Сборка train_messages/val_messages для Colab"
    )
    parser.add_argument(
        "--with-counter",
        action="store_true",
        help="добавить контрпримеры из data/counter_examples.jsonl",
    )
    parser.add_argument(
        "--val", type=int, default=5, help="размер val (из пингвиньих примеров)"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pingvin = load_jsonl(DATA_DIR / "train.jsonl")
    bad = [ex["instruction"][:50] for ex in pingvin if style_score(ex["output"]) != 5]
    assert not bad, f"у пингвиньих примеров должны быть все 5 якорей, нарушают: {bad}"

    counters: list[dict] = []
    if args.with_counter:
        counters = load_jsonl(DATA_DIR / "counter_examples.jsonl")
        leaky = [
            ex["instruction"][:50] for ex in counters if style_score(ex["output"]) != 0
        ]
        assert not leaky, (
            f"в контрпримерах не должно быть якорей стиля, нарушают: {leaky}"
        )

    rng = random.Random(args.seed)
    shuffled = pingvin.copy()
    rng.shuffle(shuffled)
    val_rows, train_rows = shuffled[: args.val], shuffled[args.val :] + counters
    rng.shuffle(
        train_rows
    )  # контрпримеры перемешаны с пингвиньими, а не приклеены хвостом

    save_jsonl([to_messages(r) for r in train_rows], DATA_DIR / "train_messages.jsonl")
    save_jsonl([to_messages(r) for r in val_rows], DATA_DIR / "val_messages.jsonl")

    steps_per_epoch = -(-len(train_rows) // 8)  # эффективный батч 8 (4 x grad_accum 2)
    print(
        f"Пингвин: {len(pingvin)} примеров (5/5 якорей у всех); контрпримеров: {len(counters)} (0/5 у всех)"
    )
    print(f"train: {len(train_rows)} диалогов -> data/train_messages.jsonl")
    print(f"val:   {len(val_rows)} диалогов -> data/val_messages.jsonl")
    print(
        f"шагов на эпоху при батче 8: {steps_per_epoch}; 8 эпох = {steps_per_epoch * 8} шагов"
    )
    print(
        "отложенный экзамен: data/test_questions.jsonl + общие вопросы в scripts/acceptance.py"
    )


if __name__ == "__main__":
    main()
