"""Вывод расхода токенов и облачной цены — ЗАДАНИЕ 1 (вывод), бонус: rich.

Важное решение: статистика печатается в **stderr**, а не в stdout. Так вывод
команд остаётся чистым для пайпов — `git diff --staged | llm commit --stats`
по-прежнему отдаёт в stdout только сообщение коммита, а таблица токенов уходит
в stderr и на пайп не влияет.
"""

from __future__ import annotations

from openai.types import CompletionUsage
from rich.console import Console
from rich.table import Table

from .pricing import PRICES_USD_PER_1M, USD_RUB, request_cost_usd

# Единая консоль для служебного вывода — всегда в stderr (см. docstring модуля).
err = Console(stderr=True)


def render_usage(usage: CompletionUsage | None, *, title: str = "расход токенов") -> None:
    """Печатает таблицу: токены запроса + во что он обошёлся бы в облаке (урок 1.3)."""
    if usage is None:
        err.print("[yellow]usage недоступен — модель не вернула счётчики токенов[/yellow]")
        return

    pin, pout, total = usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
    table = Table(
        title=f"{title}: {pin} in + {pout} out = {total} токенов  ·  локально бесплатно",
        title_style="bold cyan",
        header_style="dim",
    )
    table.add_column("если бы облако (урок 1.3)")
    table.add_column("$ / запрос", justify="right")
    table.add_column("₽ / 1000 запросов", justify="right")
    # от дешёвых к дорогим — сразу видно разброс между «мини» и флагманами
    for model in sorted(PRICES_USD_PER_1M, key=lambda m: request_cost_usd(m, pin, pout)):
        cost = request_cost_usd(model, pin, pout)
        table.add_row(model, f"${cost:.6f}", f"{cost * USD_RUB * 1000:,.2f} ₽")
    err.print(table)


class UsageMeter:
    """Копит расход за сессию — для чата, где команд много (ЗАДАНИЕ 1 + 4)."""

    def __init__(self) -> None:
        self.prompt = 0
        self.completion = 0
        self.turns = 0

    def add(self, usage: CompletionUsage | None) -> None:
        if usage is None:
            return
        self.prompt += usage.prompt_tokens
        self.completion += usage.completion_tokens
        self.turns += 1

    def render_total(self) -> None:
        total = self.prompt + self.completion
        # диапазон облачной цены всей сессии: самый дешёвый и самый дорогой класс
        costs = {m: request_cost_usd(m, self.prompt, self.completion) for m in PRICES_USD_PER_1M}
        cheap = min(costs, key=costs.get)
        dear = max(costs, key=costs.get)
        err.print(
            f"[dim]сессия: {self.turns} реплик · {self.prompt} in + {self.completion} out "
            f"= {total} токенов · в облаке ~${costs[cheap]:.5f} ({cheap}) … "
            f"${costs[dear]:.5f} ({dear})[/dim]"
        )
