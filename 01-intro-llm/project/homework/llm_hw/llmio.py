"""Обёртки над API: стриминг и разовый вызов, оба возвращают usage — ЗАДАНИЕ 1.

Тело ответа модели всегда идёт в stdout обычным print (чтобы пайпы получали
чистый текст), а служебное (счётчики, предупреждения) — забота вызывающего кода
и уходит в stderr.
"""

from __future__ import annotations

import sys

from openai.types import CompletionUsage

from .config import MODEL, client


def stream_answer(
    messages: list[dict], *, temperature: float = 0.7, max_tokens: int = 500
) -> tuple[str, CompletionUsage | None, bool]:
    """Печатает ответ токен за токеном в stdout; возвращает (текст, usage, обрезан?).

    `stream_options={"include_usage": True}` — без него при стриминге usage не
    приходит (ЗАДАНИЕ 1). Счётчики Ollama кладёт в финальный служебный чанк,
    у которого `choices` пустой, а `usage` заполнен.
    """
    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        stream=True,
        temperature=temperature,
        max_tokens=max_tokens,
        stream_options={"include_usage": True},
    )
    parts: list[str] = []
    usage: CompletionUsage | None = None
    truncated = False
    for chunk in stream:
        if chunk.usage is not None:  # финальный чанк несёт только usage
            usage = chunk.usage
        if not chunk.choices:  # у usage-чанка choices пустой — контента нет
            continue
        choice = chunk.choices[0]
        delta = choice.delta.content or ""
        if delta:
            parts.append(delta)
            sys.stdout.write(delta)
            sys.stdout.flush()
        if choice.finish_reason == "length":
            truncated = True
    sys.stdout.write("\n")
    sys.stdout.flush()
    return "".join(parts), usage, truncated


def complete(
    messages: list[dict],
    *,
    temperature: float = 0.7,
    max_tokens: int = 500,
    response_format: dict | None = None,
) -> tuple[str, CompletionUsage | None, bool]:
    """Разовый (не стриминговый) вызов; возвращает (текст, usage, обрезан?).

    Нужен там, где ответ надо получить целиком до вывода: `classify` (валидация
    JSON) и `commit`/`translate` (чистый текст в stdout, возможно, в пайп).
    """
    kwargs: dict = dict(
        model=MODEL, messages=messages, temperature=temperature, max_tokens=max_tokens
    )
    if response_format is not None:
        kwargs["response_format"] = response_format
    response = client.chat.completions.create(**kwargs)
    choice = response.choices[0]
    return choice.message.content, response.usage, choice.finish_reason == "length"
