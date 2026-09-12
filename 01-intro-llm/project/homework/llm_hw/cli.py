#!/usr/bin/env python3
"""llm — терминальный помощник на локальной Ollama (typer + rich).

Эталонное решение «Заданий для прокачки» урока 1.8. Базовые пять команд
(health, ask, summarize, classify, chat) плюс доработки домашнего задания:

    --stats     токены + облачная цена после любой команды   (задание 1)
    commit      сообщение коммита из `git diff --staged`      (задание 2)
    translate   перевод с фиксированным глоссарием терминов   (задание 3)
    chat        теперь помнит диалог между запусками           (задание 4)
    llm ...     ставится как команда через pyproject.toml      (задание 5)
"""

from __future__ import annotations

import subprocess
import sys
from typing import Optional

import httpx
import typer
from openai import APIConnectionError
from pydantic import BaseModel, Field, ValidationError
from rich.console import Console

from .config import BASE_URL, MODEL, client, ollama_root
from .history import (
    HISTORY_PATH,
    MAX_HISTORY,
    clear_history,
    load_history,
    save_history,
)
from .llmio import complete, stream_answer
from .stats import UsageMeter, err, render_usage

# --- UTF-8 для консоли и пайпов (Windows) — до создания rich-консолей ---------
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")

out = Console()  # человекочитаемый вывод в stdout (для интерактивных команд)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="rich",
    help="Терминальный помощник на локальной LLM (Ollama). "
    "Конфигурация — переменные окружения LLM_BASE_URL, LLM_MODEL, LLM_TIMEOUT_S.",
)

# Общая опция --stats: одинаковая во всех командах (задание 1 — «везде»).
StatsOpt = typer.Option(
    False, "--stats", help="расход токенов + облачная цена (урок 1.3)"
)


# --- вспомогательное чтение ввода --------------------------------------------
def read_input_text(path: str | None) -> str:
    """Текст из файла-аргумента или из stdin (для summarize/translate)."""
    if path:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    if sys.stdin.isatty():
        err.print("Передайте текст аргументом или через stdin", style="red")
        raise typer.Exit(2)
    return sys.stdin.read()


# --- health: три уровня проверки из урока 1.5 --------------------------------
@app.command()
def health() -> None:
    """Проверить сервер, модель и генерацию (урок 1.5)."""
    root = ollama_root()
    try:
        httpx.get(root, timeout=5)
    except (httpx.ConnectError, httpx.TimeoutException):
        out.print(
            f"[red][FAIL][/red] сервер {root} не отвечает — запустите Ollama (ollama serve)"
        )
        raise typer.Exit(1)
    out.print(f"[green][ OK ][/green] сервер отвечает: {root}")

    try:
        tags = httpx.get(f"{root}/api/tags", timeout=5).json()
        names = [m["name"] for m in tags.get("models") or []]
    except Exception as e:  # health обязан доложить, а не упасть
        out.print(f"[red][FAIL][/red] не удалось получить список моделей: {e}")
        raise typer.Exit(1)
    if MODEL not in names:
        out.print(
            f"[red][FAIL][/red] модель {MODEL} не скачана — выполните: ollama pull {MODEL}"
        )
        raise typer.Exit(1)
    out.print(f"[green][ OK ][/green] модель {MODEL} на месте")

    try:
        client.chat.completions.create(
            model=MODEL, max_tokens=3, messages=[{"role": "user", "content": "ping"}]
        )
    except Exception as e:
        out.print(f"[red][FAIL][/red] генерация не работает: {type(e).__name__}: {e}")
        raise typer.Exit(1)
    out.print("[green][ OK ][/green] генерация работает — всё здорово")


# --- ask: разовый вопрос со стримингом (уроки 1.2, 1.4) ----------------------
SYSTEM_ASK = (
    "Ты — лаконичный помощник разработчика. Отвечай по-русски, "
    "по существу, без лишних преамбул."
)


@app.command()
def ask(
    question: str,
    temperature: float = typer.Option(0.7, "-t", "--temperature"),
    max_tokens: int = typer.Option(500, "--max-tokens"),
    no_stream: bool = typer.Option(False, "--no-stream", help="дождаться всего ответа"),
    stats: bool = StatsOpt,
) -> None:
    """Задать один вопрос (по умолчанию ответ печатается стримингом)."""
    messages = [
        {"role": "system", "content": SYSTEM_ASK},
        {"role": "user", "content": question},
    ]
    if no_stream:
        text, usage, truncated = complete(
            messages, temperature=temperature, max_tokens=max_tokens
        )
        print(text)
    else:
        text, usage, truncated = stream_answer(
            messages, temperature=temperature, max_tokens=max_tokens
        )
    if truncated:
        err.print("[dim][ответ обрезан по --max-tokens][/dim]")
    if stats:
        render_usage(usage)


# --- summarize: файл или stdin, с защитой контекста (уроки 1.1, 1.5) ---------
MAX_INPUT_CHARS = 6000  # рабочий контекст Ollama ограничен — режем вход сами


@app.command()
def summarize(
    path: Optional[str] = typer.Argument(
        None, help="путь к файлу; если нет — читаем stdin"
    ),
    sentences: int = typer.Option(3, "-s", "--sentences"),
    stats: bool = StatsOpt,
) -> None:
    """Краткое содержание файла или stdin."""
    text = read_input_text(path)
    if len(text) > MAX_INPUT_CHARS:
        err.print(
            f"[dim][вход обрезан до {MAX_INPUT_CHARS} символов — иначе промпт молча "
            f"порежет сама Ollama, урок 1.5][/dim]"
        )
        text = text[:MAX_INPUT_CHARS]
    messages = [
        {
            "role": "system",
            "content": (
                f"Сожми текст пользователя в {sentences} предложения по-русски. "
                "Передавай только факты из текста, ничего не выдумывай."
            ),
        },
        {
            "role": "user",
            "content": f"<text>\n{text}\n</text>",
        },  # данные в разделителях, урок 1.2
    ]
    _, usage, _ = stream_answer(messages, temperature=0.3, max_tokens=300)
    if stats:
        render_usage(usage)


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


@app.command()
def classify(
    text: str,
    json_out: bool = typer.Option(False, "--json", help="вывести сырой JSON"),
    stats: bool = StatsOpt,
) -> None:
    """Определить категорию обращения в поддержку (few-shot + JSON + Pydantic)."""
    messages = [
        {"role": "system", "content": SYSTEM_CLASSIFY},
        {"role": "user", "content": f"Обращение: {text}"},
    ]
    route: TicketRoute | None = None
    usage = None
    for attempt in (1, 2):  # одна повторная попытка с текстом ошибки — урок 1.2
        raw, usage, _ = complete(
            messages,
            temperature=0.0,
            max_tokens=150,
            response_format={"type": "json_object"},
        )
        try:
            route = TicketRoute.model_validate_json(raw)
            break
        except ValidationError as e:
            if attempt == 2:
                err.print(f"[red]Модель не вернула валидный JSON:[/red] {raw!r}")
                raise typer.Exit(1)
            messages += [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": f"Исправь JSON. Ошибки валидации: {e}"},
            ]

    if json_out:
        print(route.model_dump_json())
    else:
        out.print(f"[bold]категория:[/bold] {route.category}")
        out.print(f"[bold]почему:[/bold]    {route.reason}")
    if stats:
        render_usage(usage)


# --- commit: сообщение коммита из staged-диффа — ЗАДАНИЕ 2 --------------------
COMMIT_FEW_SHOT = """Примеры (дифф → сообщение):

--- дифф ---
+++ b/auth/login.py
- def login(user): ...
+ def login(user, remember=False): ...
--- сообщение ---
feat(auth): добавь параметр remember в login

--- дифф ---
+++ b/README.md
- pip install foo
+ pip install foo bar
--- сообщение ---
docs: обнови команду установки зависимостей

--- дифф ---
+++ b/api/orders.py
-    total = sum(items)
+    total = sum(i.price for i in items)
--- сообщение ---
fix(orders): считай сумму по цене позиций, а не по объектам"""

SYSTEM_COMMIT = (
    "Ты пишешь сообщения git-коммитов по стилю Conventional Commits. Правила:\n"
    "- первая строка: `тип(область): суть` в повелительном наклонении, по-русски, ≤ 72 символов;\n"
    "- тип из набора: feat, fix, docs, refactor, test, chore, perf;\n"
    "- если изменение большое — добавь тело через пустую строку, по пунктам;\n"
    "- опиши ЧТО и ЗАЧЕМ, а не пересказывай дифф построчно;\n"
    "- верни ТОЛЬКО текст сообщения, без markdown-ограждений и пояснений.\n\n"
    + COMMIT_FEW_SHOT
)

MAX_DIFF_CHARS = 6000  # большие диффы рвут контекст — режем, как и остальной вход


def _git_staged_diff() -> str:
    """Фолбэк, если дифф не пришёл через пайп: спросим git сами."""
    try:
        return subprocess.run(
            ["git", "diff", "--staged"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        ).stdout
    except FileNotFoundError:
        return ""


@app.command()
def commit(stats: bool = StatsOpt) -> None:
    """Сгенерировать сообщение коммита из staged-диффа.

    Основной способ — пайп: [bold]git diff --staged | llm commit[/bold].
    Если пайпа нет, утилита сама выполнит `git diff --staged`.
    Чистый stdout можно сразу отдать git: [bold]... | git commit -F -[/bold].
    """
    diff = "" if sys.stdin.isatty() else sys.stdin.read()
    if not diff.strip():
        diff = _git_staged_diff()
    if not diff.strip():
        err.print("[yellow]Нет staged-изменений. Сначала: git add <файлы>[/yellow]")
        raise typer.Exit(1)
    if len(diff) > MAX_DIFF_CHARS:
        err.print(f"[dim][дифф обрезан до {MAX_DIFF_CHARS} символов][/dim]")
        diff = diff[:MAX_DIFF_CHARS]

    messages = [
        {"role": "system", "content": SYSTEM_COMMIT},
        {"role": "user", "content": f"<diff>\n{diff}\n</diff>"},
    ]
    text, usage, _ = complete(messages, temperature=0.2, max_tokens=200)
    print(text.strip())  # чистый stdout -> можно | git commit -F -
    if stats:
        render_usage(usage)


# --- translate: перевод с глоссарием — ЗАДАНИЕ 3 -----------------------------
# Глоссарий фиксирует «наши» варианты терминов: «тикет», а не «билет».
GLOSSARY = {
    "ticket": "тикет (не «билет»)",
    "issue": "тикет / issue (не «проблема», когда речь о трекере)",
    "commit": "коммит",
    "merge": "мёрж (слияние)",
    "pull request": "пул-реквест",
    "deploy": "деплой",
    "build": "сборка",
    "release": "релиз",
    "rollback": "откат",
}

# Few-shot закрепляет глоссарий на примерах — по паре под каждое направление.
FEW_SHOT_TO_RU = [
    (
        "Please create a support ticket for the failed nightly build.",
        "Пожалуйста, заведи тикет в поддержку на упавшую ночную сборку.",
    ),
    (
        "Merge the pull request after code review and deploy to staging.",
        "Влей пул-реквест после код-ревью и задеплой на стейджинг.",
    ),
]
FEW_SHOT_TO_EN = [
    (
        "Заведи тикет на баг с оплатой и назначь на бэкенд-команду.",
        "Create a ticket for the payment bug and assign it to the backend team.",
    ),
    (
        "После мёржа пул-реквеста прогони пайплайн и выкати релиз.",
        "After merging the pull request, run the pipeline and ship the release.",
    ),
]

LANG_NAMES = {"ru": "русский язык", "en": "английский язык"}


@app.command()
def translate(
    text: Optional[str] = typer.Argument(None, help="текст; если нет — читаем stdin"),
    to: str = typer.Option("ru", "--to", help="язык перевода: ru или en"),
    stats: bool = StatsOpt,
) -> None:
    """Перевести текст, соблюдая глоссарий терминов (few-shot фиксирует варианты)."""
    if to not in LANG_NAMES:
        err.print(f"[red]--to принимает ru или en, получено: {to!r}[/red]")
        raise typer.Exit(2)
    source = text if text is not None else read_input_text(None)

    glossary_lines = "\n".join(f"- {en} → {ru}" for en, ru in GLOSSARY.items())
    system = {
        "role": "system",
        "content": (
            f"Ты — профессиональный технический переводчик. Переведи текст "
            f"пользователя на {LANG_NAMES[to]}. Соблюдай глоссарий терминов "
            f"(варианты слева — эквиваленты справа):\n{glossary_lines}\n"
            "Сохраняй код, имена и форматирование. Верни только перевод, без пояснений."
        ),
    }
    shots = FEW_SHOT_TO_RU if to == "ru" else FEW_SHOT_TO_EN
    messages: list[dict] = [system]
    for src, dst in shots:  # few-shot примерами закрепляем глоссарий (урок 1.7)
        messages.append({"role": "user", "content": src})
        messages.append({"role": "assistant", "content": dst})
    messages.append({"role": "user", "content": source})

    _, usage, _ = stream_answer(messages, temperature=0.2, max_tokens=500)
    if stats:
        render_usage(usage)


# --- chat: REPL с ПАМЯТЬЮ МЕЖДУ ЗАПУСКАМИ — ЗАДАНИЕ 4 ------------------------
SYSTEM_CHAT = {
    "role": "system",
    "content": "Ты — дружелюбный ассистент разработчика. Отвечай по-русски и кратко.",
}


@app.command()
def chat(
    temperature: float = typer.Option(0.7, "-t", "--temperature"),
    fresh: bool = typer.Option(False, "--fresh", help="начать с чистой истории"),
    stats: bool = StatsOpt,
) -> None:
    """Интерактивный диалог; история переживает перезапуск (~/.llm_history.json)."""
    if fresh:
        clear_history()
        history: list[dict] = []
    else:
        history = load_history()  # обрезка до MAX_HISTORY уже внутри

    meter = UsageMeter()
    out.print(
        f"Чат с [bold]{MODEL}[/bold]. Команды: /bye — выход, /clear — забыть контекст."
    )
    if history:
        out.print(f"[dim]загружено {len(history)} сообщений из {HISTORY_PATH}[/dim]")

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
            clear_history()
            out.print("[dim][контекст очищен][/dim]")
            continue

        history.append({"role": "user", "content": user_text})
        history[:] = history[-MAX_HISTORY:]
        text, usage, _ = stream_answer(
            [SYSTEM_CHAT] + history, temperature=temperature, max_tokens=500
        )
        history.append({"role": "assistant", "content": text})
        history[:] = history[-MAX_HISTORY:]
        save_history(history)  # <- диалог сохраняется после каждой реплики (задание 4)
        if stats:
            meter.add(usage)
            render_usage(usage, title="эта реплика")
            meter.render_total()

    save_history(history)
    out.print("Пока!")


# --- точка входа для entry point (задание 5) ---------------------------------
def main() -> None:
    """Обёртка для консольной команды `llm`: одна дружелюбная точка отказа (урок 1.4)."""
    try:
        app()
    except APIConnectionError:
        err.print(f"[red]Не удалось подключиться к {BASE_URL}[/red]")
        err.print("Ollama запущена? Диагностика: [bold]llm health[/bold]")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
