#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""llm-cli — терминальный LLM-помощник на локальной Ollama.

Финальный проект модуля 1 курса «LLM для Python-разработчиков».
Здесь работает всё, что мы прошли:

  - конфигурация из окружения, OpenAI-совместимый API      (уроки 1.3, 1.4)
  - обработка ошибок и ретраи                              (урок 1.4)
  - стриминг ответа                                        (урок 1.4)
  - параметры генерации, JSON-режим, Pydantic-валидация    (урок 1.2)
  - healthcheck сервера/модели/генерации                   (урок 1.5)
  - few-shot блоком и учёт контекстного окна               (уроки 1.7, 1.1)

Примеры:
    python llm_cli.py health
    python llm_cli.py ask "Что такое GIL в Python?"
    python llm_cli.py summarize README.md
    type log.txt | python llm_cli.py summarize     # Windows (в bash — cat)
    python llm_cli.py classify "Не приходит письмо для сброса пароля"
    python llm_cli.py chat
"""

from __future__ import annotations

import argparse
import os
import sys

import httpx
from openai import APIConnectionError, OpenAI
from pydantic import BaseModel, Field, ValidationError

# --- Windows-консоль и пайпы: печатаем в UTF-8, чтобы кириллица не ломалась --
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")

# --- Конфигурация из окружения — как DSN базы данных (урок 1.4) -------------
BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
MODEL = os.getenv("LLM_MODEL", "qwen2.5:3b")
API_KEY = os.getenv("LLM_API_KEY", "ollama")  # локально не проверяется
TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "120"))

client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
    timeout=TIMEOUT_S,
    max_retries=2,  # транзиентные ошибки (сеть, 429, 5xx) SDK ретраит сам
)


def ollama_root() -> str:
    """http://host:11434/v1 -> http://host:11434 — корень родного API Ollama."""
    return BASE_URL.rsplit("/v1", 1)[0]


# --- health: трёхуровневая проверка из урока 1.5 ----------------------------
def cmd_health(args: argparse.Namespace) -> int:
    root = ollama_root()
    try:
        httpx.get(root, timeout=5)
    except (httpx.ConnectError, httpx.TimeoutException):
        print(f"[FAIL] сервер {root} не отвечает — запустите Ollama (ollama serve)")
        return 1
    print(f"[ OK ] сервер отвечает: {root}")

    try:
        tags = httpx.get(f"{root}/api/tags", timeout=5).json()
        names = [m["name"] for m in tags.get("models") or []]
    except Exception as e:  # health обязан доложить, а не упасть
        print(f"[FAIL] не удалось получить список моделей: {e}")
        return 1
    if MODEL not in names:
        print(f"[FAIL] модель {MODEL} не скачана — выполните: ollama pull {MODEL}")
        return 1
    print(f"[ OK ] модель {MODEL} на месте")

    try:
        client.chat.completions.create(
            model=MODEL,
            max_tokens=3,
            messages=[{"role": "user", "content": "ping"}],
        )
    except Exception as e:
        print(f"[FAIL] генерация не работает: {type(e).__name__}: {e}")
        return 1
    print("[ OK ] генерация работает — всё здорово")
    return 0


# --- ask: разовый вопрос со стримингом (урок 1.4) ----------------------------
SYSTEM_ASK = (
    "Ты — лаконичный помощник разработчика. Отвечай по-русски, "
    "по существу, без лишних преамбул."
)


def stream_answer(messages: list, *, temperature: float, max_tokens: int) -> str:
    """Печатает ответ токен за токеном, возвращает полный текст."""
    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        stream=True,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    parts: list[str] = []
    for chunk in stream:
        if not chunk.choices:  # служебные чанки без содержимого
            continue
        delta = chunk.choices[0].delta.content or ""
        parts.append(delta)
        print(delta, end="", flush=True)
    print()
    return "".join(parts)


def cmd_ask(args: argparse.Namespace) -> int:
    messages = [
        {"role": "system", "content": SYSTEM_ASK},
        {"role": "user", "content": args.question},
    ]
    if args.no_stream:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        choice = response.choices[0]
        print(choice.message.content)
        if choice.finish_reason == "length":  # урок 1.4: не молчим про обрезку
            print("[ответ обрезан по --max-tokens]", file=sys.stderr)
        if args.stats and response.usage:
            print(
                f"[токены: {response.usage.prompt_tokens} in / "
                f"{response.usage.completion_tokens} out]",
                file=sys.stderr,
            )
    else:
        stream_answer(
            messages, temperature=args.temperature, max_tokens=args.max_tokens
        )
    return 0


# --- summarize: файл или stdin, с уважением к контексту (уроки 1.1, 1.5) ----
MAX_INPUT_CHARS = 6000  # рабочий контекст Ollama ограничен — режем вход сами


def read_input_text(path: str | None) -> str:
    if path:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    if sys.stdin.isatty():
        print("Передайте файл аргументом или текст через stdin", file=sys.stderr)
        raise SystemExit(2)
    return sys.stdin.read()


def cmd_summarize(args: argparse.Namespace) -> int:
    text = read_input_text(args.path)
    if len(text) > MAX_INPUT_CHARS:
        print(
            f"[вход обрезан до {MAX_INPUT_CHARS} символов — иначе промпт молча "
            f"порежет сама Ollama, урок 1.5]",
            file=sys.stderr,
        )
        text = text[:MAX_INPUT_CHARS]
    messages = [
        {
            "role": "system",
            "content": (
                f"Сожми текст пользователя в {args.sentences} предложения "
                "по-русски. Передавай только факты из текста, ничего не выдумывай."
            ),
        },
        # данные всегда в разделителях — урок 1.2
        {"role": "user", "content": f"<text>\n{text}\n</text>"},
    ]
    stream_answer(messages, temperature=0.3, max_tokens=300)
    return 0


# --- classify: few-shot блоком + JSON + Pydantic (уроки 1.7, 1.2) ------------
FEW_SHOT_BLOCK = """Примеры:
Обращение: Не могу оплатить картой, платёж отклоняется -> {"category": "оплата", "reason": "проблема с платежом"}
Обращение: Кнопка оплаты не нажимается, в консоли ошибка JS -> {"category": "техника", "reason": "сломан интерфейс продукта"}
Обращение: Как поменять email в настройках профиля? -> {"category": "аккаунт", "reason": "управление профилем"}
Обращение: Хотим купить рекламу на вашем сайте -> {"category": "другое", "reason": "не запрос в поддержку"}"""

SYSTEM_CLASSIFY = (
    "Ты — маршрутизатор обращений в службу поддержки. Определи категорию "
    "обращения (оплата, техника, аккаунт или другое) и кратко объясни почему. "
    "Отвечай строго JSON-объектом с полями category и reason.\n\n" + FEW_SHOT_BLOCK
)


class TicketRoute(BaseModel):
    category: str = Field(pattern="^(оплата|техника|аккаунт|другое)$")
    reason: str


def cmd_classify(args: argparse.Namespace) -> int:
    messages = [
        {"role": "system", "content": SYSTEM_CLASSIFY},
        {"role": "user", "content": f"Обращение: {args.text}"},
    ]
    route = None
    for attempt in (1, 2):  # одна повторная попытка с текстом ошибки — урок 1.2
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.0,
            max_tokens=150,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content
        try:
            route = TicketRoute.model_validate_json(raw)
            break
        except ValidationError as e:
            if attempt == 2:
                print(f"Модель не вернула валидный JSON: {raw!r}", file=sys.stderr)
                return 1
            messages += [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": f"Исправь JSON. Ошибки валидации: {e}"},
            ]
    if args.json:
        print(route.model_dump_json())
    else:
        print(f"категория: {route.category}")
        print(f"почему:    {route.reason}")
    return 0


# --- chat: REPL с памятью и обрезкой истории (уроки 1.1, 1.6) ----------------
MAX_HISTORY = 12  # держим последние N сообщений — простейшая обрезка контекста


def cmd_chat(args: argparse.Namespace) -> int:
    system = {
        "role": "system",
        "content": "Ты — дружелюбный ассистент разработчика. Отвечай по-русски и кратко.",
    }
    history: list[dict] = []
    print(f"Чат с {MODEL}. Команды: /bye — выход, /clear — забыть контекст.")
    while True:
        try:
            user_text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_text:
            continue
        if user_text in ("/bye", "/exit", "/quit"):
            break
        if user_text == "/clear":
            history.clear()
            print("[контекст очищен]")
            continue
        history.append({"role": "user", "content": user_text})
        if len(history) > MAX_HISTORY:
            # чат «помнит» ровно то, что мы отправляем, — урок 1.1 и решение 2
            # урока 1.6: память диалога живёт в промпте, а не в модели
            history[:] = history[-MAX_HISTORY:]
        answer = stream_answer(
            [system] + history, temperature=args.temperature, max_tokens=500
        )
        history.append({"role": "assistant", "content": answer})
    print("Пока!")
    return 0


# --- точка входа --------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="llm",
        description="Терминальный помощник на локальной LLM (Ollama). "
        "Конфигурация: переменные окружения LLM_BASE_URL, LLM_MODEL, LLM_TIMEOUT_S.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("health", help="проверить сервер, модель и генерацию")
    p.set_defaults(func=cmd_health)

    p = sub.add_parser("ask", help="задать один вопрос")
    p.add_argument("question")
    p.add_argument("-t", "--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=500)
    p.add_argument("--no-stream", action="store_true", help="дождаться всего ответа")
    p.add_argument(
        "--stats", action="store_true", help="расход токенов (вместе с --no-stream)"
    )
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("summarize", help="краткое содержание файла или stdin")
    p.add_argument("path", nargs="?", help="путь к файлу; если нет — читаем stdin")
    p.add_argument("-s", "--sentences", type=int, default=3)
    p.set_defaults(func=cmd_summarize)

    p = sub.add_parser("classify", help="категория обращения в поддержку")
    p.add_argument("text")
    p.add_argument("--json", action="store_true", help="вывести сырой JSON")
    p.set_defaults(func=cmd_classify)

    p = sub.add_parser("chat", help="интерактивный диалог с памятью")
    p.add_argument("-t", "--temperature", type=float, default=0.7)
    p.set_defaults(func=cmd_chat)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except APIConnectionError:
        print(f"Не удалось подключиться к {BASE_URL}", file=sys.stderr)
        print("Ollama запущена? Диагностика: python llm_cli.py health", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
