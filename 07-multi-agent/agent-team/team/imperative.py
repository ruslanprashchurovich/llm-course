"""Оркестратор урока 7.2 на чистом Python — эталон поведения для графа задания 1.

Логика повторяет ``run_team`` из ноутбука 02-orchestrator.ipynb; отличие одно:
LLM и судья приходят снаружи (внедрение зависимостей), поэтому тот же код гоняется
и на Ollama, и на ScriptedLLM в тестах.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from team import prompts
from team.judge import TestJudge, Verdict, extract_code
from team.llm import ChatLLM
from team.log import Message, TeamLog
from team.prompts import Role
from team.tasks import CodingTask

VERDICT_ACCEPTED = "ACCEPTED"
VERDICT_ACCEPTED_ESCALATED = "ACCEPTED (после эскалации)"
VERDICT_NO_CODE = "REJECTED: разработчик не вернул код"
VERDICT_HUMAN = "REJECTED: нужен человек"


@dataclass
class Outcome:
    verdict: str
    code: str
    report: str
    llm_calls: int
    prompt_tokens: int
    answer_tokens: int
    seconds: float
    log: TeamLog
    escalated: bool = False

    @property
    def accepted(self) -> bool:
        return self.verdict.startswith("ACCEPTED")


def ask(llm: ChatLLM, log: TeamLog, role: Role, user: str, recipient: str, kind: str) -> str:
    """Один вызов роли: system роли, её бюджет, запись в журнал (Agent.ask из 7.2)."""
    reply = llm.chat(role.system, user, num_predict=role.num_predict)
    log.add(Message(sender=role.name, recipient=recipient, kind=kind, content=reply.content,
                    prompt_tokens=reply.prompt_tokens, answer_tokens=reply.answer_tokens,
                    seconds=reply.seconds))
    return reply.content


def judge_and_log(judge: TestJudge, code: str, log: TeamLog) -> Verdict:
    verdict = judge(code)
    log.add(Message("тесты", "оркестратор", "test_report", verdict.report))
    if log.verbose:
        print(f"    тесты: {verdict.passed}/{verdict.total}")
    return verdict


def run_team(task: CodingTask, llm: ChatLLM, judge: TestJudge, *,
             max_fix_rounds: int = 2, log: TeamLog | None = None) -> Outcome:
    """Конвейер -> фикс-цикл -> эскалация с чистого листа -> фикс-цикл -> вердикт."""
    log = log if log is not None else TeamLog()
    started = time.perf_counter()
    escalated = False

    def finish(verdict: str, code: str, report: str) -> Outcome:
        p_tok, a_tok = log.total_tokens()
        return Outcome(verdict, code, report, log.llm_calls(), p_tok, a_tok,
                       round(time.perf_counter() - started, 1), log, escalated)

    def fix_cycle(code: str, verdict: Verdict) -> tuple[str, Verdict, str]:
        """До max_fix_rounds раундов «трассировка -> правка -> тесты»."""
        trace = ""
        for _ in range(max_fix_rounds):
            if verdict.ok:
                break
            first_fail = verdict.first_fail
            assert first_fail is not None
            trace = ask(llm, log, prompts.TRACER,
                        prompts.tracer_prompt(code, verdict.report, first_fail),
                        "разработчик", "trace")
            fixed = ask(llm, log, prompts.DEVELOPER,
                        prompts.fix_prompt(task.text, code, verdict.report, trace),
                        "тесты", "code")
            code = extract_code(fixed) or code
            verdict = judge_and_log(judge, code, log)
        return code, verdict, trace

    # --- конвейер -------------------------------------------------------------
    spec = ask(llm, log, prompts.ANALYST, prompts.analyst_prompt(task.text),
               "разработчик", "spec")
    dev_answer = ask(llm, log, prompts.DEVELOPER, prompts.developer_prompt(task.text, spec),
                     "тесты", "code")
    code = extract_code(dev_answer)
    if code is None:                       # агент вернул прозу вместо кода
        return finish(VERDICT_NO_CODE, "", "")
    verdict = judge_and_log(judge, code, log)

    # --- фикс-цикл v1 (лимит! - рифма max_steps из урока 3.5) ------------------
    code, verdict, trace = fix_cycle(code, verdict)
    if verdict.ok:
        return finish(VERDICT_ACCEPTED, code, verdict.report)

    # --- эскалация: чистый лист, диагноз есть, старого кода НЕТ ----------------
    escalated = True
    if log.verbose:
        print("    оркестратор: фикс-цикл исчерпан -> эскалация (чистый лист)")
    rebuilt = ask(llm, log, prompts.DEVELOPER,
                  prompts.rebuild_prompt(task.text, verdict.report, trace), "тесты", "code")
    code = extract_code(rebuilt) or code
    verdict = judge_and_log(judge, code, log)

    # --- новый код заслуживает свой фикс-цикл -----------------------------------
    code, verdict, _ = fix_cycle(code, verdict)
    if verdict.ok:
        return finish(VERDICT_ACCEPTED_ESCALATED, code, verdict.report)
    return finish(VERDICT_HUMAN, code, verdict.report)
