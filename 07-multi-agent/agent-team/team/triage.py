"""Задание 3 (капстоун): команда против одиночки на триаже тикетов «Пингвин.Хост».

Задача из практики поддержки (урок 7.1, «тикет-крик»): по тексту тикета выдать
JSON с категорией, приоритетом, дословными фактами, флагом «нужен человек» и
резюме. Судья — код, три проверки:

* **схема**  — JSON парсится и проходит pydantic-модель ``Triage`` (extra=forbid);
* **заземление** — каждая цитата в ``facts`` действительно есть в тикете
  (аналитик 7.1 выдумал требование — здесь это ловится кодом);
* **правила** — политика триажа как консистентность полей (outage не бывает low,
  high без конкретных фактов — крик, возврат денег и любой high — только человек, ...).

Отчёт судьи читается строками ``[PASS]/[FAIL] проверка: детали`` — это интерфейс
фикс-цикла (урок 7.2). Три конфигурации гоняются одним eval'ом:

* ``solo``      — один вызов, судья только оценивает;
* ``solo+fix``  — одиночка + фикс-цикл по отчёту судьи (та же обвязка, что у команды);
* ``team``      — экстрактор фактов -> код отбрасывает незаземлённые цитаты ->
                  триажёр -> судья -> фикс-цикл -> «нужен человек».

Разница solo+fix vs team — это цена и польза РАЗДЕЛЕНИЯ РОЛЕЙ, разница solo vs
solo+fix — польза судьи-кода. Окупилась ли команда — смотрите в таблицу, а не в
ожидания.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from team.llm import ChatLLM
from team.log import Message, TeamLog
from team.prompts import Role

CATEGORIES = ("outage", "billing", "access", "performance", "question", "other")
PRIORITIES = ("low", "normal", "high", "critical")
PRIORITY_RANK = {p: i for i, p in enumerate(PRIORITIES)}

Category = Literal["outage", "billing", "access", "performance", "question", "other"]
Priority = Literal["low", "normal", "high", "critical"]


class Triage(BaseModel):
    """Контракт ответа — та же строгость, что в схемах модуля 8."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    category: Category
    priority: Priority
    facts: list[str] = Field(min_length=1, max_length=5)
    needs_human: bool
    summary: str = Field(min_length=5, max_length=200)


SCHEMA_TEXT = """{"category": "outage | billing | access | performance | question | other",
 "priority": "low | normal | high | critical",
 "facts": ["дословная цитата из тикета", "ещё одна (всего 1-4)"],
 "needs_human": true | false,
 "summary": "одно предложение своими словами, до 200 символов"}"""

POLICY_TEXT = """Правила триажа «Пингвин.Хост».
Категории: outage — сайт, почта или база клиента недоступны или деградируют; billing — счета,
списания, тарифы, возвраты; access — вход в панель, пароли, FTP, взлом; performance — работает,
но медленно; question — как сделать / сколько стоит / есть ли услуга; other — всё остальное
(удаление аккаунта, юридические запросы, крик без фактов).
Приоритет:
- outage — только high или critical; critical (SEV1) — недоступно всё или для всех, либо взлом.
- critical допустим только для outage и access.
- high и critical требуют КОНКРЕТНЫХ фактов в facts: домен, код ошибки, время, номер счёта,
  имя сервиса или файла. Крик без фактов — category other, priority low, попросить детали.
- billing — low или normal: деньги важны, но сервис работает.
- question — low или normal.
needs_human = true, если тикет не закрыть автоответом: любой outage, любой high/critical,
взлом или блокировка аккаунта, возврат денег или двойное списание, удаление аккаунта и
другие юридические запросы. needs_human = false для вопросов, консультаций и замедлений
без ошибок.
facts — ДОСЛОВНЫЕ цитаты из тикета, 1-4 штуки; ничего не выдумывать и не пересказывать.
summary — одно предложение своими словами."""


# ------------------------------------------------------------------ eval-набор
@dataclass(frozen=True)
class Ticket:
    id: str
    text: str
    category: str
    priority: str
    needs_human: bool
    gold_fact: str  # дословная цитата — доказательство, что заземлённый ответ возможен
    note: str = ""


TICKETS: list[Ticket] = [
    Ticket(
        "T01",
        "С 09:40 сайт pingvin-shop.ru отдаёт 502 Bad Gateway всем посетителям. "
        "Магазин стоит, теряем заказы каждую минуту.",
        "outage",
        "critical",
        True,
        "отдаёт 502 Bad Gateway всем посетителям",
        "SEV1: всё и для всех",
    ),
    Ticket(
        "T02",
        "С карты списали 1990 рублей дважды за один тариф «Старт»: в кабинете два "
        "одинаковых счёта № 48211 и № 48212. Верните лишнее списание.",
        "billing",
        "normal",
        True,
        "два одинаковых счёта № 48211 и № 48212",
        "возврат — человек",
    ),
    Ticket(
        "T03",
        "Подскажите, как подключить бесплатный SSL-сертификат Let's Encrypt к домену "
        "через панель? Нашёл только платный вариант.",
        "question",
        "low",
        False,
        "как подключить бесплатный SSL-сертификат Let's Encrypt",
    ),
    Ticket(
        "T04",
        "Не могу войти в панель управления: пароль верный, но выводится «аккаунт "
        "заблокирован». Хостинг оплачен до декабря, сайты работают.",
        "access",
        "normal",
        True,
        "выводится «аккаунт заблокирован»",
        "разблокирует оператор",
    ),
    Ticket(
        "T05",
        "Последние три дня сайт грузится 8–10 секунд вместо привычных двух, особенно "
        "вечером. Тариф «Бизнес», трафик не менялся.",
        "performance",
        "normal",
        False,
        "сайт грузится 8–10 секунд вместо привычных двух",
    ),
    Ticket(
        "T06",
        "ВСЁ СЛОМАЛОСЬ!!! НИЧЕГО НЕ РАБОТАЕТ!!! СРОЧНО!!!",
        "other",
        "low",
        False,
        "ВСЁ СЛОМАЛОСЬ",
        "тикет-крик из 7.1: фактов нет",
    ),
    Ticket(
        "T07",
        "Почта на домене pingvin-cafe.ru не принимает письма с 14:00: отправители "
        "получают ошибку 550 relay denied. Сайт при этом работает.",
        "outage",
        "high",
        True,
        "получают ошибку 550 relay denied",
        "SEV2: деградация функции",
    ),
    Ticket(
        "T08",
        "Пришёл счёт за продление домена на 2027 год, хотя мы уже оплатили продление "
        "в июле, чек № 77120 сохранился. Это дубль или что-то новое?",
        "billing",
        "low",
        False,
        "чек № 77120 сохранился",
        "вопрос по счёту, без возврата",
    ),
    Ticket(
        "T09",
        "Кто-то сменил пароль от FTP, а в корне сайта появился файл wp-shell.php, "
        "который мы не создавали. Похоже, нас взломали.",
        "access",
        "critical",
        True,
        "появился файл wp-shell.php",
        "взлом = SEV1",
    ),
    Ticket(
        "T10",
        "Планируем переезд с другого хостинга. Есть ли у вас инструмент переноса сайта "
        "на WordPress и сколько стоит услуга переноса?",
        "question",
        "low",
        False,
        "сколько стоит услуга переноса",
    ),
    Ticket(
        "T11",
        "Раз в час база MySQL перестаёт отвечать на 30–40 секунд, в логах ошибка "
        "Too many connections. Сайт в это время показывает ошибку 500.",
        "performance",
        "high",
        True,
        "перестаёт отвечать на 30–40 секунд",
    ),
    Ticket(
        "T12",
        "Прошу удалить мой аккаунт и все персональные данные по 152-ФЗ. Договор № 5531, "
        "услуги больше не нужны.",
        "other",
        "normal",
        True,
        "удалить мой аккаунт и все персональные данные по 152-ФЗ",
        "юридический запрос — человек",
    ),
]


# ------------------------------------------------------------------ судья-код
@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class TriageVerdict:
    triage: Triage | None
    checks: list[Check]

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)

    def passed(self, name: str) -> bool:
        return any(c.name == name and c.ok for c in self.checks)

    @property
    def report(self) -> str:
        return "\n".join(
            f"[{'PASS' if c.ok else 'FAIL'}] {c.name}: {c.detail}" for c in self.checks
        )


_QUOTES = str.maketrans({ch: "" for ch in "«»\"'“”„`"})
_DASHES = str.maketrans({"–": "-", "—": "-"})
CONCRETE = re.compile(
    r"\d|\.(ru|com|io|net|org|рф|php|html|js)\b|сервер|сайт|домен|баз[аы]|бэкап|счёт|счет|api|ssl|"
    r"почт|dns|диск|ftp|mysql|панел|логин|парол|shell|файл|ошибк|error|denied|gateway|timeout",
    re.IGNORECASE,
)
REFUND = re.compile(r"дважды|двойн|верн[иу]|возврат|лишнее списание", re.IGNORECASE)
JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def normalize(text: str) -> str:
    """Регистр, ё/е, кавычки, тире и пробелы не считаются: цитата — это смысл, не байты."""
    text = text.lower().replace("ё", "е").translate(_QUOTES).translate(_DASHES)
    return re.sub(r"\s+", " ", text).strip(" .,;:!?-")


def is_grounded(fact: str, ticket_text: str) -> bool:
    fact_n = normalize(fact)
    return len(fact_n) >= 4 and fact_n in normalize(ticket_text)


def parse_triage(raw: str) -> tuple[Triage | None, str]:
    """JSON -> Triage; при ошибке — читаемая причина для фикс-цикла."""
    match = JSON_OBJECT.search(raw)
    if not match:
        return None, "в ответе нет JSON-объекта {...}"
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return None, f"JSON не парсится: {exc.msg} (позиция {exc.pos})"
    try:
        return Triage.model_validate(data), "схема соблюдена"
    except ValidationError as exc:
        issues = [
            f"{'.'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']}"
            for e in exc.errors()[:3]
        ]
        return None, "схема нарушена — " + "; ".join(issues)


def policy_violations(triage: Triage, ticket_text: str) -> list[str]:
    """Консистентность полей по правилам триажа. Не решает задачу — проверяет ответ."""
    issues: list[str] = []
    if triage.category == "outage" and triage.priority in ("low", "normal"):
        issues.append(
            "R1: outage не бывает low/normal — недоступность клиента это high или critical"
        )
    if triage.priority == "critical" and triage.category not in ("outage", "access"):
        issues.append(
            "R2: critical допустим только для outage (всё лежит) и access (взлом)"
        )
    if triage.priority in ("high", "critical") and not any(
        CONCRETE.search(f) for f in triage.facts
    ):
        issues.append(
            "R3: priority high/critical, но в facts нет конкретики "
            "(домен, код ошибки, время, номер, имя сервиса) — крик без фактов это low"
        )
    if (
        triage.category == "billing"
        and REFUND.search(ticket_text)
        and not triage.needs_human
    ):
        issues.append(
            "R4: возврат денег / двойное списание — needs_human должен быть true"
        )
    if triage.category == "question" and (
        triage.priority in ("high", "critical") or triage.needs_human
    ):
        issues.append("R5: question — это low/normal и needs_human = false")
    if (
        triage.category == "outage" or triage.priority in ("high", "critical")
    ) and not triage.needs_human:
        issues.append(
            "R6: outage и любой high/critical не закрыть автоответом — needs_human должен быть true"
        )
    if triage.category == "billing" and triage.priority in ("high", "critical"):
        issues.append("R7: billing — это low/normal: деньги важны, но сервис работает")
    return issues


class TriageJudge:
    """Три проверки кодом: схема, заземление, правила. Ноль LLM."""

    __test__ = False

    def __call__(self, ticket_text: str, raw_answer: str) -> TriageVerdict:
        triage, detail = parse_triage(raw_answer)
        checks = [Check("schema", triage is not None, detail)]
        if triage is None:
            return TriageVerdict(None, checks)

        missing = [f for f in triage.facts if not is_grounded(f, ticket_text)]
        checks.append(
            Check(
                "grounded",
                not missing,
                "все цитаты найдены в тикете"
                if not missing
                else "цитат нет в тикете (выдумано или пересказано): "
                + "; ".join(f"«{m}»" for m in missing),
            )
        )
        issues = policy_violations(triage, ticket_text)
        checks.append(
            Check(
                "rules",
                not issues,
                "правила триажа соблюдены" if not issues else " | ".join(issues),
            )
        )
        return TriageVerdict(triage, checks)


# ------------------------------------------------------------------ роли и промпты
EXTRACTOR = Role(
    "extractor",
    "экстрактор",
    "Ты — аналитик поддержки «Пингвин.Хост». Твоя единственная задача — выписать из тикета "
    "ДОСЛОВНЫЕ цитаты с фактами: что сломано, где, с какого времени, коды ошибок, номера "
    "счетов и договоров. Ничего не выдумывай и не пересказывай своими словами. "
    "Верни ТОЛЬКО JSON-массив строк из 1-4 элементов.",
    200,
)

TRIAGER = Role(
    "triager",
    "триажёр",
    "Ты — триажёр тикетов поддержки хостинга «Пингвин.Хост». Верни ТОЛЬКО один JSON-объект "
    f"по схеме, без пояснений и без markdown:\n{SCHEMA_TEXT}\n\n{POLICY_TEXT}",
    300,
)

SOLO = Role(
    "solo",
    "одиночка",
    "Ты — триажёр тикетов поддержки хостинга «Пингвин.Хост». Сам выпиши дословные факты "
    "и классифицируй тикет. Верни ТОЛЬКО один JSON-объект по схеме, без пояснений и без "
    f"markdown:\n{SCHEMA_TEXT}\n\n{POLICY_TEXT}",
    300,
)

TRIAGE_ROLES = {role.key: role for role in (EXTRACTOR, TRIAGER, SOLO)}
TRIAGE_ROLE_OF_SYSTEM = {role.system: role.key for role in TRIAGE_ROLES.values()}


def extractor_prompt(ticket_text: str) -> str:
    return f"Тикет:\n{ticket_text}"


def triager_prompt(ticket_text: str, facts: list[str]) -> str:
    facts_json = json.dumps(facts, ensure_ascii=False)
    return (
        f"Тикет:\n{ticket_text}\n\nФакты, выписанные аналитиком и проверенные по тексту "
        f"(используй их в facts ДОСЛОВНО, ничего не добавляй):\n{facts_json}"
    )


def solo_prompt(ticket_text: str) -> str:
    return f"Тикет:\n{ticket_text}"


def fix_prompt(ticket_text: str, previous: str, report: str) -> str:
    return (
        f"Тикет:\n{ticket_text}\n\nТвой предыдущий ответ:\n{previous}\n\n"
        f"Проверка кодом нашла ошибки:\n{report}\n\n"
        "Исправь ровно то, что помечено FAIL, и верни ТОЛЬКО JSON."
    )


def parse_facts(raw: str) -> list[str]:
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    return [str(item) for item in data if isinstance(item, str) and item.strip()][:4]


# ------------------------------------------------------------------ прогоны
@dataclass
class TriageOutcome:
    ticket: Ticket
    mode: str
    verdict: TriageVerdict
    accepted: bool
    escalated: bool
    llm_calls: int
    prompt_tokens: int
    answer_tokens: int
    seconds: float
    facts_dropped: int = 0  # незаземлённые цитаты, отброшенные кодом между ролями
    degraded_calls: int = 0  # ответы от запасной модели прокси (капстоун 5.7)

    @property
    def triage(self) -> Triage | None:
        return self.verdict.triage

    def correct(self, field_name: str) -> bool:
        if self.triage is None:
            return False
        return getattr(self.triage, field_name) == getattr(self.ticket, field_name)

    @property
    def priority_within_one(self) -> bool:
        if self.triage is None:
            return False
        return (
            abs(
                PRIORITY_RANK[self.triage.priority]
                - PRIORITY_RANK[self.ticket.priority]
            )
            <= 1
        )


class _Run:
    """Учёт одного прогона: вызовы, токены, деградации, журнал."""

    def __init__(self, llm: ChatLLM, log: TeamLog) -> None:
        self.llm, self.log = llm, log
        self.calls = self.p_tok = self.a_tok = self.degraded = 0
        self.started = time.perf_counter()

    def ask(self, role: Role, user: str, recipient: str, kind: str) -> str:
        reply = self.llm.chat(role.system, user, num_predict=role.num_predict)
        self.calls += 1
        self.p_tok += reply.prompt_tokens
        self.a_tok += reply.answer_tokens
        self.degraded += bool(reply.degraded_to)
        self.log.add(
            Message(
                role.name,
                recipient,
                kind,
                reply.content,
                reply.prompt_tokens,
                reply.answer_tokens,
                reply.seconds,
            )
        )
        return reply.content

    def judge(self, judge: TriageJudge, ticket: Ticket, raw: str) -> TriageVerdict:
        verdict = judge(ticket.text, raw)
        self.log.add(Message("судья", "оркестратор", "check_report", verdict.report))
        return verdict

    def outcome(
        self,
        ticket: Ticket,
        mode: str,
        verdict: TriageVerdict,
        *,
        facts_dropped: int = 0,
    ) -> TriageOutcome:
        return TriageOutcome(
            ticket,
            mode,
            verdict,
            accepted=verdict.ok,
            escalated=not verdict.ok,
            llm_calls=self.calls,
            prompt_tokens=self.p_tok,
            answer_tokens=self.a_tok,
            seconds=round(time.perf_counter() - self.started, 1),
            facts_dropped=facts_dropped,
            degraded_calls=self.degraded,
        )


def _fix_loop(
    run: _Run,
    judge: TriageJudge,
    ticket: Ticket,
    role: Role,
    raw: str,
    verdict: TriageVerdict,
    fix_rounds: int,
) -> TriageVerdict:
    for _ in range(fix_rounds):
        if verdict.ok:
            break
        raw = run.ask(
            role, fix_prompt(ticket.text, raw, verdict.report), "судья", "json"
        )
        verdict = run.judge(judge, ticket, raw)
    return verdict


def run_solo(
    ticket: Ticket,
    llm: ChatLLM,
    judge: TriageJudge,
    *,
    fix_rounds: int = 0,
    log: TeamLog | None = None,
) -> TriageOutcome:
    """Одиночка: один промпт делает всё; fix_rounds=0 — судья только оценивает."""
    log = log if log is not None else TeamLog(verbose=False)
    run = _Run(llm, log)
    raw = run.ask(SOLO, solo_prompt(ticket.text), "судья", "json")
    verdict = run.judge(judge, ticket, raw)
    verdict = _fix_loop(run, judge, ticket, SOLO, raw, verdict, fix_rounds)
    return run.outcome(ticket, "solo" if fix_rounds == 0 else "solo+fix", verdict)


def run_team(
    ticket: Ticket,
    llm: ChatLLM,
    judge: TriageJudge,
    *,
    fix_rounds: int = 1,
    log: TeamLog | None = None,
) -> TriageOutcome:
    """Команда: экстрактор -> фильтр кодом -> триажёр -> судья -> фикс-цикл -> человек."""
    log = log if log is not None else TeamLog(verbose=False)
    run = _Run(llm, log)

    raw_facts = run.ask(EXTRACTOR, extractor_prompt(ticket.text), "триажёр", "facts")
    facts = parse_facts(raw_facts)
    grounded = [f for f in facts if is_grounded(f, ticket.text)]
    dropped = len(facts) - len(grounded)
    if dropped and log.verbose:
        print(f"    оркестратор: отброшено незаземлённых цитат — {dropped}")
    if not grounded:  # экстрактор не справился — триажёр цитирует сам
        grounded = []

    raw = run.ask(TRIAGER, triager_prompt(ticket.text, grounded), "судья", "json")
    verdict = run.judge(judge, ticket, raw)
    verdict = _fix_loop(run, judge, ticket, TRIAGER, raw, verdict, fix_rounds)
    return run.outcome(ticket, "team", verdict, facts_dropped=dropped)


MODES = {
    "solo": lambda t, llm, judge, log: run_solo(t, llm, judge, fix_rounds=0, log=log),
    "solo+fix": lambda t, llm, judge, log: run_solo(
        t, llm, judge, fix_rounds=1, log=log
    ),
    "team": lambda t, llm, judge, log: run_team(t, llm, judge, fix_rounds=1, log=log),
}


# ------------------------------------------------------------------ сводка eval'а
@dataclass
class EvalSummary:
    mode: str
    outcomes: list[TriageOutcome] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.outcomes)

    def share(self, predicate) -> float:
        return sum(1 for o in self.outcomes if predicate(o)) / self.n if self.n else 0.0

    @property
    def accepted(self) -> float:
        return self.share(lambda o: o.accepted)

    @property
    def schema_ok(self) -> float:
        return self.share(lambda o: o.verdict.passed("schema"))

    @property
    def grounded_ok(self) -> float:
        return self.share(lambda o: o.verdict.passed("grounded"))

    @property
    def rules_ok(self) -> float:
        return self.share(lambda o: o.verdict.passed("rules"))

    @property
    def category_acc(self) -> float:
        return self.share(lambda o: o.correct("category"))

    @property
    def priority_acc(self) -> float:
        return self.share(lambda o: o.correct("priority"))

    @property
    def priority_within_one(self) -> float:
        return self.share(lambda o: o.priority_within_one)

    @property
    def needs_human_acc(self) -> float:
        return self.share(lambda o: o.correct("needs_human"))

    @property
    def quality(self) -> float:
        """Одно число для сравнения: среднее трёх точностей по всем тикетам."""
        return (self.category_acc + self.priority_acc + self.needs_human_acc) / 3

    @property
    def llm_calls(self) -> int:
        return sum(o.llm_calls for o in self.outcomes)

    @property
    def tokens(self) -> int:
        return sum(o.prompt_tokens + o.answer_tokens for o in self.outcomes)

    @property
    def seconds(self) -> float:
        return sum(o.seconds for o in self.outcomes)

    @property
    def escalations(self) -> int:
        return sum(1 for o in self.outcomes if o.escalated)

    @property
    def degraded_calls(self) -> int:
        return sum(o.degraded_calls for o in self.outcomes)


def evaluate(
    mode: str,
    tickets: list[Ticket],
    llm: ChatLLM,
    judge: TriageJudge,
    *,
    verbose: bool = True,
) -> EvalSummary:
    runner = MODES[mode]
    summary = EvalSummary(mode)
    for ticket in tickets:
        log = TeamLog(verbose=False)
        outcome = runner(ticket, llm, judge, log)
        summary.outcomes.append(outcome)
        if verbose:
            got = outcome.triage
            got_text = (
                f"{got.category}/{got.priority}/{'human' if got.needs_human else 'bot'}"
                if got
                else "— (схема)"
            )
            flag = "OK " if outcome.accepted else "ЧЕЛ"
            marks = "".join(
                "+" if outcome.correct(f) else "-"
                for f in ("category", "priority", "needs_human")
            )
            print(
                f"  [{mode:8}] {ticket.id} {flag} {got_text:28} gold "
                f"{ticket.category}/{ticket.priority}/{'human' if ticket.needs_human else 'bot'}"
                f"  [{marks}] {outcome.llm_calls} выз., {outcome.seconds:.0f} c"
            )
    return summary


def compare_table(summaries: list[EvalSummary]) -> str:
    rows = [
        ("тикетов", lambda s: str(s.n)),
        ("принято судьёй", lambda s: f"{s.accepted:.0%}"),
        ("эскалаций «нужен человек»", lambda s: str(s.escalations)),
        ("схема соблюдена", lambda s: f"{s.schema_ok:.0%}"),
        ("цитаты заземлены", lambda s: f"{s.grounded_ok:.0%}"),
        ("правила соблюдены", lambda s: f"{s.rules_ok:.0%}"),
        ("category верно", lambda s: f"{s.category_acc:.0%}"),
        ("priority верно (точно)", lambda s: f"{s.priority_acc:.0%}"),
        ("priority верно (±1 уровень)", lambda s: f"{s.priority_within_one:.0%}"),
        ("needs_human верно", lambda s: f"{s.needs_human_acc:.0%}"),
        ("КАЧЕСТВО (среднее трёх)", lambda s: f"{s.quality:.0%}"),
        ("вызовов LLM", lambda s: str(s.llm_calls)),
        ("токенов всего", lambda s: str(s.tokens)),
        ("секунд", lambda s: f"{s.seconds:.0f}"),
    ]
    if any(s.degraded_calls for s in summaries):
        rows.append(("ответов от запасной модели", lambda s: str(s.degraded_calls)))
    head = ["метрика"] + [s.mode for s in summaries]
    body = [[name] + [fn(s) for s in summaries] for name, fn in rows]
    widths = [max(len(r[i]) for r in [head, *body]) for i in range(len(head))]
    fmt = lambda r: "  ".join(c.ljust(widths[i]) for i, c in enumerate(r))  # noqa: E731
    return "\n".join([fmt(head), "  ".join("-" * w for w in widths), *map(fmt, body)])


def paid_off(team: EvalSummary, baseline: EvalSummary) -> str:
    """Честный вердикт по цифрам: качество против цены."""
    dq = (team.quality - baseline.quality) * 100
    cost = team.tokens / baseline.tokens if baseline.tokens else float("inf")
    if dq <= 0:
        return (
            f"Команда НЕ окупилась против «{baseline.mode}»: качество {dq:+.0f} п.п. "
            f"при цене ×{cost:.1f} по токенам."
        )
    return (
        f"Команда дала {dq:+.0f} п.п. качества за ×{cost:.1f} токенов против "
        f"«{baseline.mode}» — окупилось, если эти пункты стоят удвоенного счёта за LLM."
    )
