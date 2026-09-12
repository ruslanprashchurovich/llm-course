"""Санитизация пользовательского ввода (урок 3.7).

Три вердикта:
  clean      - пропустить;
  suspicious - пропустить, но пометить (паттерн-детекторы ошибаются в обе стороны);
  rejected   - отбить до конвейера (грубое нарушение формы).

Детектор по чёрному списку - ранний дешёвый рубеж, а не стена: настоящая
защита от прорвавшихся инъекций - grounding-промпт, порог и канонический
отказ (урок 3.4).
"""

from __future__ import annotations

import re

INJECTION_PATTERNS = [
    r"забудь\s+(все\s+)?(правила|инструкции)",
    r"игнорируй\s+(предыдущие|все)\s+(правила|инструкции)",
    r"новые?\s+инструкци",
    r"ты\s+больше\s+не\s+ассистент",
    r"ignore\s+(all\s+)?(previous\s+)?instructions",
    r"disregard\s+(the\s+)?(rules|instructions)",
    r"system\s*prompt",
]
# побег из разделителей нашего промпта (<context>, <quote> - уроки 3.1/3.4)
TAG_PATTERN = re.compile(r"</?\s*context\s*>|</?\s*quote\s*>", re.IGNORECASE)
MIN_LEN, MAX_LEN = 3, 500


def sanitize_question(raw: str) -> dict:
    question = " ".join(str(raw).split())

    if not (MIN_LEN <= len(question) <= MAX_LEN):
        return {"verdict": "rejected", "reason": "length",
                "question": question[:MAX_LEN]}

    neutralized = TAG_PATTERN.sub("", question)
    tags_stripped = neutralized != question

    hits = [p for p in INJECTION_PATTERNS
            if re.search(p, neutralized, re.IGNORECASE)]
    if hits or tags_stripped:
        return {"verdict": "suspicious",
                "reason": "tags" if tags_stripped else "pattern",
                "patterns": hits, "question": neutralized}

    return {"verdict": "clean", "question": neutralized}
