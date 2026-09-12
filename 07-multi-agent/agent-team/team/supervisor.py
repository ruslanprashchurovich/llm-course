"""Задание 2: supervisor по-настоящему — AutoGen SelectorGroupChat и цена выбора.

Три исполнителя (analyst, developer, critic), следующего выбирает LLM. Цену
выбора считаем в цифрах: сколько токенов уходит на сами решения «кто говорит
следующим», сколько раз selector переспрашивали (модель не назвала ровно одно имя)
и как часто выбор расходится с конвейером урока 7.2 — ``expected_next`` и есть
эталонная политика оркестратора, записанная как функция.

Три режима для сравнения:

* ``selector``   — выбирает LLM (supervisor-паттерн урока 7.1);
* ``policy``     — та же SelectorGroupChat, но ``selector_func=expected_next``:
                   ноль токенов на выбор, ноль ошибок — оркестратор 7.2 в одежде AutoGen;
* ``roundrobin`` — фиксированный круг: бесплатно, но после критика говорит аналитик,
                   а не разработчик.

Критик — LLM (как в задании) или судья-код (``critic="tests"``): участник чата,
который вместо рассуждений гоняет приёмочные тесты. Так проверяем тезис урока 7.4
«в чате некуда воткнуть судью-код» — оказывается, можно: судьёй становится участник.

Чисто питоновская часть (политика, подсчёт) не импортирует AutoGen — тесты
гоняют её без фреймворка; сам чат живёт в ``run_supervisor``.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Any

from team.config import Settings
from team.judge import TestJudge, extract_code
from team.tasks import CodingTask

PARTICIPANTS = ("analyst", "developer", "critic")
APPROVE = "ВЕРДИКТ_ПРИНЯТО"      # стоп-слово: не подстрока отрицания (частая ошибка 1 урока 7.4)
REWORK = "ВЕРДИКТ_ДОРАБОТКА"

DESCRIPTIONS = {
    "analyst": "аналитик: выписывает требования и крайние случаи к задаче; НЕ пишет код",
    "developer": "разработчик: пишет функцию Python по требованиям или исправляет её по замечаниям критика",
    "critic": "критик: проверяет последний код разработчика и выносит вердикт "
              f"{APPROVE} или {REWORK}",
}

SELECTOR_PROMPT = f"""Ты — руководитель команды разработки. Участники и их роли:
{{roles}}

Порядок работы команды: сначала analyst выписывает требования, затем developer пишет код,
затем critic проверяет; если critic вынес {REWORK} — снова говорит developer;
если critic вынес {APPROVE} — работа закончена.

Прочитай переписку и выбери, кто говорит следующим, из списка {{participants}}.
Ответь ТОЛЬКО одним именем из списка, без пояснений.

{{history}}
"""

CRITIC_SYSTEM = f"""Ты — критик кода. Проверь ПОСЛЕДНИЙ код разработчика по чеклисту:
1. Функция называется как в задаче и принимает нужные аргументы.
2. Каждый пример из задачи даст указанный результат — подставь входы мысленно и запиши результат.
3. Ошибочные входы поднимают ValueError, как требует задача.
Если все три пункта выполнены — ответь ровно одним словом: {APPROVE}.
Иначе ответь {REWORK} и перечисли, что именно исправить (по одной строке на пункт)."""


# ------------------------------------------------------------------ политика и подсчёт
def expected_next(sources: list[str], contents: list[str]) -> str | None:
    """Кого ДОЛЖЕН выбрать supervisor по конвейеру 7.2. None = разговор должен закончиться."""
    last_source, last_content = sources[-1], contents[-1]
    if last_source == "user":
        return "analyst"
    if last_source == "analyst":
        return "developer"
    if last_source == "developer":
        return "critic"
    if last_source == "critic":
        return None if APPROVE in last_content else "developer"
    return None


@dataclass
class SelectionRecord:
    """Один вызов selector-LLM: что ответил и почём."""
    content: str
    prompt_tokens: int
    completion_tokens: int
    seconds: float = 0.0


@dataclass
class SelectionStats:
    mode: str
    speakers: list[str]                       # порядок говорящих без стартового user
    turns: int
    selector_calls: int
    selector_prompt_tokens: int
    selector_completion_tokens: int
    worker_prompt_tokens: int
    worker_completion_tokens: int
    wrong_picks: list[tuple[int, str, str]] = field(default_factory=list)  # (ход, ожидали, выбрали)
    stop_reason: str | None = None
    tests_passed: int | None = None
    tests_total: int | None = None
    seconds: float = 0.0

    @property
    def retries(self) -> int:
        """Переспросы selector'а: вызовов больше, чем ходов."""
        return max(self.selector_calls - self.turns, 0)

    @property
    def selector_tokens(self) -> int:
        return self.selector_prompt_tokens + self.selector_completion_tokens

    @property
    def worker_tokens(self) -> int:
        return self.worker_prompt_tokens + self.worker_completion_tokens

    @property
    def selector_share(self) -> float:
        total = self.selector_tokens + self.worker_tokens
        return self.selector_tokens / total if total else 0.0

    @property
    def wrong_share(self) -> float:
        return len(self.wrong_picks) / self.turns if self.turns else 0.0

    def table(self) -> str:
        tests = (f"{self.tests_passed}/{self.tests_total}"
                 if self.tests_total is not None else "-")
        rows = [
            ("режим", self.mode),
            ("ходов (реплик агентов)", str(self.turns)),
            ("порядок говорящих", " -> ".join(self.speakers) or "-"),
            ("вызовов selector-LLM", f"{self.selector_calls} (переспросов: {self.retries})"),
            ("токены selector (prompt+answer)",
             f"{self.selector_prompt_tokens}+{self.selector_completion_tokens} = {self.selector_tokens}"),
            ("токены исполнителей", f"{self.worker_prompt_tokens}+{self.worker_completion_tokens}"
                                    f" = {self.worker_tokens}"),
            ("доля токенов на выбор", f"{self.selector_share:.0%}"),
            ("выбор не по конвейеру", f"{len(self.wrong_picks)}/{self.turns} ({self.wrong_share:.0%})"),
            ("остановка", str(self.stop_reason)),
            ("приёмка последнего кода", tests),
            ("время", f"{self.seconds:.0f} c"),
        ]
        width = max(len(k) for k, _ in rows)
        return "\n".join(f"{k.ljust(width)}  {v}" for k, v in rows)


def score_selection(mode: str, messages: list[tuple[str, str]], records: list[SelectionRecord],
                    worker_usage: tuple[int, int], *, stop_reason: str | None = None,
                    seconds: float = 0.0) -> SelectionStats:
    """Сверяет фактический порядок говорящих с политикой конвейера и сводит цену."""
    sources = [src for src, _ in messages]
    contents = [txt for _, txt in messages]
    wrong: list[tuple[int, str, str]] = []
    for i in range(1, len(messages)):                 # messages[0] — задача от user
        expected = expected_next(sources[:i], contents[:i])
        if expected is not None and sources[i] != expected:
            wrong.append((i, expected, sources[i]))
    return SelectionStats(
        mode=mode,
        speakers=sources[1:],
        turns=len(messages) - 1,
        selector_calls=len(records),
        selector_prompt_tokens=sum(r.prompt_tokens for r in records),
        selector_completion_tokens=sum(r.completion_tokens for r in records),
        worker_prompt_tokens=worker_usage[0],
        worker_completion_tokens=worker_usage[1],
        wrong_picks=wrong,
        stop_reason=stop_reason,
        seconds=seconds,
    )


def last_code_block(messages: list[tuple[str, str]]) -> str | None:
    """Последний блок кода от developer — его и принимаем (или нет)."""
    for source, content in reversed(messages):
        if source == "developer":
            code = extract_code(content)
            if code:
                return code
    return None


def code_critic_reply(code: str | None, judge: TestJudge) -> str:
    """Реплика критика-судьи: отчёт тестов + стоп-слово. Никакой LLM."""
    if code is None:
        return f"{REWORK}\nВ переписке нет блока кода ```python``` от developer."
    verdict = judge(code)
    if verdict.ok:
        return f"{APPROVE}\nПриёмка {verdict.passed}/{verdict.total}:\n{verdict.report}"
    return f"{REWORK}\nПриёмка {verdict.passed}/{verdict.total}:\n{verdict.report}"


# ------------------------------------------------------------------ AutoGen
def run_supervisor(settings: Settings, task: CodingTask, judge: TestJudge, *,
                   mode: str = "selector", critic: str = "llm", max_messages: int = 10,
                   verbose: bool = True) -> SelectionStats:
    """Собирает команду AutoGen и возвращает статистику цены выбора."""
    return asyncio.run(_run_supervisor_async(settings, task, judge, mode=mode, critic=critic,
                                             max_messages=max_messages, verbose=verbose))


async def _run_supervisor_async(settings: Settings, task: CodingTask, judge: TestJudge, *,
                                mode: str, critic: str, max_messages: int,
                                verbose: bool) -> SelectionStats:
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
    from autogen_agentchat.teams import RoundRobinGroupChat, SelectorGroupChat

    from team import prompts

    if mode not in {"selector", "policy", "roundrobin"}:
        raise ValueError(f"mode={mode!r}: ожидается selector | policy | roundrobin")

    worker_client = make_client(settings, num_predict=600)
    selector_client = make_client(settings, num_predict=30)   # имя — это одно слово

    analyst = AssistantAgent(
        "analyst", model_client=worker_client, description=DESCRIPTIONS["analyst"],
        system_message=prompts.ANALYST.system + " Выпиши 4-6 требований и 3-4 крайних случая списками.")
    developer = AssistantAgent(
        "developer", model_client=worker_client, description=DESCRIPTIONS["developer"],
        system_message=prompts.DEVELOPER.system)
    if critic == "tests":
        critic_agent = make_tests_critic("critic", judge, DESCRIPTIONS["critic"])
    elif critic == "llm":
        critic_agent = AssistantAgent("critic", model_client=worker_client,
                                      description=DESCRIPTIONS["critic"], system_message=CRITIC_SYSTEM)
    else:
        raise ValueError(f"critic={critic!r}: ожидается llm | tests")

    participants = [analyst, developer, critic_agent]
    termination = TextMentionTermination(APPROVE) | MaxMessageTermination(max_messages)

    if mode == "roundrobin":
        team = RoundRobinGroupChat(participants, termination_condition=termination)
    else:
        selector_func = None
        if mode == "policy":
            def selector_func(thread):        # оркестратор 7.2 как функция выбора
                return expected_next([m.source for m in thread],
                                     [str(getattr(m, "content", "")) for m in thread])
        team = SelectorGroupChat(
            participants, model_client=selector_client, selector_prompt=SELECTOR_PROMPT,
            termination_condition=termination, allow_repeated_speaker=False,
            selector_func=selector_func,
        )

    started = time.perf_counter()
    result = await team.run(task=f"Задача:\n{task.text}")
    seconds = time.perf_counter() - started

    messages = [(str(getattr(m, "source", "?")), str(getattr(m, "content", "")))
                for m in result.messages]
    if verbose:
        for source, content in messages:
            print(f"--- {source}:\n{content[:400]}\n")
    usage = worker_client.total_usage()
    stats = score_selection(mode, messages, selector_client.records,
                            (usage.prompt_tokens, usage.completion_tokens),
                            stop_reason=result.stop_reason, seconds=seconds)
    code = last_code_block(messages)
    verdict = judge(code) if code else None
    stats.tests_passed = verdict.passed if verdict else 0
    stats.tests_total = judge.total
    await worker_client.close()
    await selector_client.close()
    return stats


def make_client(settings: Settings, *, num_predict: int):
    """OllamaChatCompletionClient со счётчиком вызовов; think — полем запроса."""
    from autogen_core.models import ModelFamily, ModelInfo
    from autogen_ext.models.ollama import OllamaChatCompletionClient

    class CountingOllamaClient(OllamaChatCompletionClient):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.records: list[SelectionRecord] = []

        async def create(self, messages, **kwargs):   # type: ignore[override]
            t0 = time.perf_counter()
            result = await super().create(messages, **kwargs)
            self.records.append(SelectionRecord(
                str(result.content), result.usage.prompt_tokens,
                result.usage.completion_tokens, time.perf_counter() - t0))
            return result

    return CountingOllamaClient(
        model=settings.model, host=settings.ollama_url,
        options={"temperature": settings.temperature, "num_predict": num_predict},
        think=settings.think,   # у qwen3.5 иначе рассуждения съедят num_predict (урок 5.2)
        model_info=ModelInfo(vision=False, function_calling=False, json_output=True,
                             family=ModelFamily.UNKNOWN, structured_output=False,
                             multiple_system_messages=False),
    )


def make_tests_critic(name: str, judge: TestJudge, description: str):
    """Критик-судья: участник чата без LLM — гоняет приёмочные тесты."""
    from autogen_agentchat.agents import BaseChatAgent
    from autogen_agentchat.base import Response
    from autogen_agentchat.messages import TextMessage

    class TestsCritic(BaseChatAgent):
        def __init__(self) -> None:
            super().__init__(name, description)
            self._seen: list[tuple[str, str]] = []

        @property
        def produced_message_types(self):
            return (TextMessage,)

        async def on_messages(self, messages, cancellation_token) -> Response:
            self._seen.extend((m.source, str(getattr(m, "content", ""))) for m in messages)
            reply = code_critic_reply(last_code_block(self._seen), judge)
            return Response(chat_message=TextMessage(content=reply, source=self.name))

        async def on_reset(self, cancellation_token) -> None:
            self._seen.clear()

    return TestsCritic()


def summarize(stats_by_mode: dict[str, SelectionStats]) -> str:
    """Сводная таблица режимов: цена выбора в одном взгляде."""
    header = ["режим", "ходов", "selector-вызовов", "selector-токенов", "доля", "не по конвейеру",
              "приёмка", "остановка"]
    rows = [header]
    for mode, s in stats_by_mode.items():
        rows.append([mode, str(s.turns), f"{s.selector_calls} (+{s.retries} переспр.)",
                     str(s.selector_tokens), f"{s.selector_share:.0%}",
                     f"{len(s.wrong_picks)}/{s.turns}",
                     f"{s.tests_passed}/{s.tests_total}", str(s.stop_reason)[:40]])
    widths = [max(len(r[i]) for r in rows) for i in range(len(header))]
    lines = ["  ".join(c.ljust(widths[i]) for i, c in enumerate(r)) for r in rows]
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(lines)
