"""Безопасность RAG: санитизация документов, детект prompt injection, маскирование PII,
проверка ответа модели.

Все функции здесь — чистые (без сети и моделей), поэтому их легко тестировать
и дёшево запускать на каждом чанке при индексации.

Модель угроз (кратко):
  1. Документ приходит из недоверенного источника (вики, тикеты, письма, парсинг сайта)
     и содержит инструкции для модели -> indirect prompt injection.
  2. Документ содержит персональные данные (ПДн) или секреты -> утечка в ответ и в логи.
  3. Модель в ответе просит перейти по ссылке или «сливает» системный промпт
     -> exfiltration через ссылку.
"""

from __future__ import annotations

import hashlib
import re
import secrets
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable

# ---------------------------------------------------------------------------
# 1. Нормализация текста
# ---------------------------------------------------------------------------

# Невидимые символы: zero-width, bidi-override, word joiner, BOM.
# Классика обхода фильтров: вставить zero-width space внутрь слова
# ("иг<U+200B>норируй все инструкции") — глазами не видно, регулярка не срабатывает.
# Диапазоны задаём кодами символов, а не самими символами: невидимые символы
# в исходнике нечитаемы и теряются при копировании кода.
_INVISIBLE_RANGES: tuple[tuple[int, int], ...] = (
    (0x00AD, 0x00AD),  # soft hyphen
    (0x200B, 0x200F),  # zero-width space/non-joiner/joiner, метки направления текста
    (0x202A, 0x202E),  # bidi override (RLO и компания)
    (0x2060, 0x2064),  # word joiner, невидимые операторы
    (0x206A, 0x206F),  # устаревшие форматирующие символы
    (0xFEFF, 0xFEFF),  # BOM / zero-width no-break space
)
INVISIBLE_RE = re.compile(
    "[" + "".join(f"{chr(lo)}-{chr(hi)}" for lo, hi in _INVISIBLE_RANGES) + "]"
)
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
HTML_TAG_RE = re.compile(r"</?[a-zA-Z][^>]{0,200}>")
MD_LINK_RE = re.compile(r"\[([^\]]{0,200})\]\((https?://[^)\s]{1,500})\)")
URL_RE = re.compile(r"https?://[^\s<>\"'\)\]]+", re.IGNORECASE)
LONG_WS_RE = re.compile(r"[ \t\f\v]{3,}")
MANY_NEWLINES_RE = re.compile(r"\n{4,}")


def normalize_text(text: str) -> str:
    """Приводит текст к каноничному виду ПЕРЕД любыми проверками.

    Порядок важен: сначала нормализуем Unicode и убираем невидимые символы,
    иначе детекторы injection легко обойти невидимым разделителем.
    """
    text = unicodedata.normalize("NFKC", text)
    text = INVISIBLE_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = LONG_WS_RE.sub(" ", text)
    text = MANY_NEWLINES_RE.sub("\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# 2. Детект prompt injection
# ---------------------------------------------------------------------------

# Каждое правило: (имя, вес, регулярка). Вес складывается в общий score.
# Это эвристика: ловит типовые атаки, но не является гарантией. Второй слой
# защиты — жёсткий системный промпт и проверка ответа (см. ниже).
INJECTION_RULES: tuple[tuple[str, int, re.Pattern[str]], ...] = (
    (
        "ignore_previous",
        4,
        re.compile(
            r"(игнорируй|забудь|не\s+обращай\s+внимания\s+на)\s+(все\s+)?"
            r"(предыдущи\w+|прошл\w+|прежни\w+|выше)\s*"
            r"(инструкц\w+|указан\w+|промпт\w*|правил\w+)?"
            r"|ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)"
            r"|disregard\s+(all\s+)?(previous|prior)\s+",
            re.IGNORECASE,
        ),
    ),
    (
        "role_override",
        3,
        re.compile(
            r"(ты\s+(больше\s+не|теперь|с\s+этого\s+момента)\s+\w+)"
            r"|(веди\s+себя\s+как)"
            r"|(you\s+are\s+(now|no\s+longer))"
            r"|(act\s+as\s+(a\s+)?(dan|developer\s+mode|jailbroken))"
            r"|(новая\s+рол[ьи])",
            re.IGNORECASE,
        ),
    ),
    (
        "fake_system_block",
        4,
        re.compile(
            r"(^|\n|<!--)\s*(system|систем\w*)\s*(:|=|>>|\]|prompt)"
            r"|<\|?(im_start|system|endoftext)\|?>"
            r"|(\[\s*(system|instructions?)\s*\])"
            r"|(###\s*(system|instruction))",
            re.IGNORECASE,
        ),
    ),
    (
        "instruction_to_assistant",
        3,
        re.compile(
            r"(инструкц\w+|указан\w+|команд\w+)\s+(для|ассистент\w*|бот\w*|модел\w+|ai|llm)"
            r"|(для\s+(ассистент\w+|бота|модели|llm))"
            r"|(instructions?\s+(for|to)\s+(the\s+)?(assistant|ai|model|bot))",
            re.IGNORECASE,
        ),
    ),
    (
        "secrecy_request",
        3,
        re.compile(
            r"(никому\s+не\s+говори|не\s+упоминай\s+(этот|это|данн\w+)|скрой\s+(этот|это)"
            r"|не\s+сообщай\s+(пользовател\w+|сотрудник\w+))"
            r"|(do\s+not\s+(tell|mention|reveal)|keep\s+this\s+secret)",
            re.IGNORECASE,
        ),
    ),
    (
        "credential_bait",
        4,
        re.compile(
            r"(универсальн\w+\s+пароль|общий\s+пароль\s+для\s+всех|master\s+password)"
            r"|(скачайте?\s+.{0,40}\.(zip|exe|msi|bat|ps1))"
            r"|(запустите?\s+.{0,30}\.(exe|bat|ps1|sh)\s+от\s+администратора)"
            r"|(отправ\w+\s+(пароль|токен|ключ|конфиг))",
            re.IGNORECASE,
        ),
    ),
    (
        "exfiltration_link",
        4,
        re.compile(
            r"https?://[^\s]{0,200}[?&][^\s]{0,50}="
            r"[^\s]{0,50}(<|%3C|\{)?\s*(систем\w+\s+промпт|system\s*prompt|"
            r"текст\s+вопроса|history|context|q=|data=)",
            re.IGNORECASE,
        ),
    ),
    (
        "prompt_leak_request",
        3,
        re.compile(
            r"(выведи|покажи|повтори|раскрой)\s+(свой\s+)?(системн\w+\s+промпт|инструкц\w+)"
            r"|(repeat|print|reveal)\s+(your\s+)?(system\s*prompt|instructions)",
            re.IGNORECASE,
        ),
    ),
    (
        "tool_abuse",
        3,
        re.compile(
            r"(вызови|выполни|запусти)\s+(инструмент|функцию|команду|http|curl)"
            r"|(call\s+the\s+tool|execute\s+the\s+following)",
            re.IGNORECASE,
        ),
    ),
    (
        "base64_blob",
        2,
        re.compile(r"\b[A-Za-z0-9+/]{120,}={0,2}\b"),
    ),
)

RISK_THRESHOLDS = {"low": 0, "medium": 3, "high": 6}


@dataclass
class InjectionReport:
    """Результат проверки текста на prompt injection."""

    score: int = 0
    matched_rules: list[str] = field(default_factory=list)
    samples: dict[str, str] = field(default_factory=dict)

    @property
    def risk(self) -> str:
        if self.score >= RISK_THRESHOLDS["high"]:
            return "high"
        if self.score >= RISK_THRESHOLDS["medium"]:
            return "medium"
        return "low"

    @property
    def is_suspicious(self) -> bool:
        return self.score > 0


def scan_injection(text: str) -> InjectionReport:
    """Ищет признаки prompt injection. Работает по нормализованному тексту."""
    report = InjectionReport()
    haystack = normalize_text(text)
    # Скрытые блоки (HTML-комментарии) — сами по себе подозрительны в контенте вики.
    hidden = HTML_COMMENT_RE.findall(haystack)
    if hidden:
        report.score += 3
        report.matched_rules.append("hidden_html_comment")
        report.samples["hidden_html_comment"] = hidden[0][:160]
    for name, weight, pattern in INJECTION_RULES:
        match = pattern.search(haystack)
        if match:
            report.score += weight
            report.matched_rules.append(name)
            report.samples[name] = match.group(0)[:160]
    return report


# ---------------------------------------------------------------------------
# 3. Санитизация чанка
# ---------------------------------------------------------------------------


@dataclass
class SanitizedChunk:
    text: str
    injection: InjectionReport
    pii_found: dict[str, int]
    removed: list[str] = field(default_factory=list)


def neutralize_urls(text: str) -> str:
    """Делает ссылки некликабельными и лишает их query-параметров.

    Так документ не может «попросить» модель вставить ссылку с утечкой данных.
    """

    def _clean(match: re.Match[str]) -> str:
        url = match.group(0)
        base = url.split("?", 1)[0].split("#", 1)[0]
        return f"[ссылка: {base.replace('://', '(:)//')}]"

    return URL_RE.sub(_clean, text)


def sanitize_document(
    text: str,
    *,
    trusted: bool,
    strip_urls: bool = False,
    mask_pii_flag: bool = True,
) -> SanitizedChunk:
    """Готовит текст документа к индексации.

    trusted=False (пользовательский контент) — удаляем скрытые блоки и теги,
    trusted=True — доверяем разметке, но всё равно нормализуем и считаем PII.
    """
    removed: list[str] = []
    clean = normalize_text(text)
    report = scan_injection(clean)

    if not trusted:
        if HTML_COMMENT_RE.search(clean):
            clean = HTML_COMMENT_RE.sub(" ", clean)
            removed.append("html_comments")
        if HTML_TAG_RE.search(clean):
            clean = HTML_TAG_RE.sub(" ", clean)
            removed.append("html_tags")
        # Markdown-ссылку разворачиваем в текст + отдельный безопасный URL.
        if MD_LINK_RE.search(clean):
            clean = MD_LINK_RE.sub(r"\1 (\2)", clean)
            removed.append("md_links")
        if strip_urls:
            clean = neutralize_urls(clean)
            removed.append("urls_neutralized")

    pii_found: dict[str, int] = {}
    if mask_pii_flag:
        clean, pii_found = mask_pii(clean)

    clean = normalize_text(clean)
    return SanitizedChunk(
        text=clean, injection=report, pii_found=pii_found, removed=removed
    )


# ---------------------------------------------------------------------------
# 4. PII и секреты
# ---------------------------------------------------------------------------

PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+\.[A-Za-zА-Яа-я]{2,}\b"),
    "PHONE_RU": re.compile(
        r"(?<!\d)(?:\+7|8)[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}(?!\d)"
    ),
    "CARD": re.compile(r"(?<!\d)(?:\d{4}[\s\-]?){3}\d{4}(?!\d)"),
    "SNILS": re.compile(r"(?<!\d)\d{3}-\d{3}-\d{3}[\s]\d{2}(?!\d)"),
    "INN": re.compile(r"(?<!\d)(?:\d{10}|\d{12})(?!\d)"),
    "PASSPORT_RU": re.compile(r"(?<!\d)\d{2}\s?\d{2}\s?\d{6}(?!\d)"),
    "IP": re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])"),
}

SECRET_PATTERNS: dict[str, re.Pattern[str]] = {
    "OPENAI_KEY": re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}\b"),
    "AWS_KEY": re.compile(r"\bAKIA[0-9A-Z]{12,}\b"),
    "BEARER": re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{16,}\b", re.IGNORECASE),
    "PASSWORD_KV": re.compile(
        r"(?i)\b(пароль|password|passwd|pwd|секрет|secret|token|api[_\-]?key)\b\s*[:=]\s*\S{4,}"
    ),
    "PRIVATE_KEY": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
}

# Приватные диапазоны IP не считаем персональными данными: это инфраструктура.
_PRIVATE_IP_PREFIXES = ("10.", "192.168.", "127.", "172.16.", "172.17.", "172.18.")


def _luhn_ok(digits: str) -> bool:
    """Проверка контрольной суммы номера карты (алгоритм Луна)."""
    nums = [int(ch) for ch in digits if ch.isdigit()]
    if len(nums) < 13:
        return False
    total = 0
    for i, digit in enumerate(reversed(nums)):
        if i % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def _pseudonym(value: str, entity: str) -> str:
    """Устойчивый псевдоним: одинаковый вход -> одинаковая метка.

    Полезно, чтобы в логах можно было сопоставить события, не храня сами ПДн.
    """
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]
    return f"[{entity}:{digest}]"


def mask_pii(text: str, *, pseudonymize: bool = True) -> tuple[str, dict[str, int]]:
    """Заменяет персональные данные на метки. Возвращает (текст, счётчики по типам)."""
    counts: dict[str, int] = {}
    result = text

    for entity, pattern in PII_PATTERNS.items():

        def _replace(match: re.Match[str], entity: str = entity) -> str:
            value = match.group(0)
            if entity == "CARD" and not _luhn_ok(value):
                return value  # похоже на номер, но контрольная сумма не сходится
            if entity == "IP" and value.startswith(_PRIVATE_IP_PREFIXES):
                return value  # внутренний адрес, не ПДн
            if entity == "INN" and len(value) == 10 and value.startswith("0"):
                return value
            counts[entity] = counts.get(entity, 0) + 1
            return _pseudonym(value, entity) if pseudonymize else f"[{entity}]"

        result = pattern.sub(_replace, result)

    return result, counts


def redact_secrets(text: str) -> tuple[str, dict[str, int]]:
    """Вырезает секреты (ключи, токены, пароли в формате key: value)."""
    counts: dict[str, int] = {}
    result = text
    for name, pattern in SECRET_PATTERNS.items():

        def _replace(match: re.Match[str], name: str = name) -> str:
            counts[name] = counts.get(name, 0) + 1
            return f"[SECRET:{name}]"

        result = pattern.sub(_replace, result)
    return result, counts


def scrub_for_logs(text: str, limit: int = 400) -> str:
    """Готовит текст к записи в лог: маскирует ПДн и секреты, обрезает длину."""
    clean, _ = mask_pii(text)
    clean, _ = redact_secrets(clean)
    clean = clean.replace("\n", " ")
    if len(clean) > limit:
        clean = clean[:limit] + "…"
    return clean


# ---------------------------------------------------------------------------
# 5. Проверка ответа модели (output guard)
# ---------------------------------------------------------------------------


def make_canary() -> str:
    """Канареечный токен для системного промпта.

    Кладём его в системный промпт; если он всплыл в ответе — промпт утёк.
    """
    return "CANARY-" + secrets.token_hex(6).upper()


CITATION_RE = re.compile(r"\[#(\d+)\]")


@dataclass
class AnswerVerdict:
    allowed: bool
    problems: list[str] = field(default_factory=list)
    citations: list[int] = field(default_factory=list)

    @property
    def reason(self) -> str:
        return ", ".join(self.problems) if self.problems else "ok"


def check_answer(
    answer: str,
    *,
    allowed_citations: Iterable[int],
    context_text: str = "",
    canary: str | None = None,
    require_citation: bool = True,
) -> AnswerVerdict:
    """Проверяет ответ модели ПЕРЕД отдачей пользователю.

    Ловит четыре класса проблем:
      * утечку системного промпта (канарейка);
      * ссылки, которых не было в контексте (exfiltration / фишинг);
      * ссылки на несуществующие источники (галлюцинация цитат);
      * ответ без цитат, когда цитаты обязательны.
    """
    problems: list[str] = []
    allowed = set(int(c) for c in allowed_citations)

    if canary and canary in answer:
        problems.append("system_prompt_leak")

    answer_urls = {u.rstrip(".,);") for u in URL_RE.findall(answer)}
    context_urls = {u.rstrip(".,);") for u in URL_RE.findall(context_text)}
    unknown = answer_urls - context_urls
    if unknown:
        problems.append(f"unknown_urls:{len(unknown)}")

    citations = [int(c) for c in CITATION_RE.findall(answer)]
    bad_citations = [c for c in citations if c not in allowed]
    if bad_citations:
        problems.append(f"invalid_citations:{sorted(set(bad_citations))}")
    if require_citation and not citations:
        problems.append("no_citation")

    leftover_secrets = redact_secrets(answer)[1]
    if leftover_secrets:
        problems.append("secret_in_answer")

    return AnswerVerdict(
        allowed=not problems, problems=problems, citations=sorted(set(citations))
    )


SAFE_FALLBACK_ANSWER = (
    "Я не могу дать ответ на этот вопрос по имеющимся документам. "
    "Похоже, в найденных материалах есть некорректные или недоверенные инструкции. "
    "Обратитесь в #it-help или #security — инцидент уже записан в логи."
)

__all__ = [
    "normalize_text",
    "scan_injection",
    "InjectionReport",
    "sanitize_document",
    "SanitizedChunk",
    "mask_pii",
    "redact_secrets",
    "scrub_for_logs",
    "neutralize_urls",
    "check_answer",
    "AnswerVerdict",
    "make_canary",
    "SAFE_FALLBACK_ANSWER",
    "PII_PATTERNS",
    "INJECTION_RULES",
]
