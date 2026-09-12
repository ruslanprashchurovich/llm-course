"""Ядро RAG-сервиса: конвейер с эшелонированной обороной (уроки 3.1-3.5).

Главное архитектурное решение - ЗАВИСИМОСТИ ПРИХОДЯТ СНАРУЖИ:
RagService не знает ни про Qdrant, ни про Ollama - только про две
async-функции: search (вопрос -> чанки со скорами) и generate
(messages, **options -> текст). Прод передаёт настоящие (app/main.py),
тесты - фейки (tests/conftest.py). Поэтому логика конвейера
тестируется без Docker и без LLM.

Рубежи (урок 3.4) и агентный ход (урок 3.5) по пути запроса:

  0. санитизация входа - до любых вызовов;
  1. порог релевантности; при промахе и agent_mode - ОДИН retry поиска
     с переформулировкой запроса моделью (синонимы, официальные термины);
  2. промпт v3 + требование дословной цитаты <quote>;
  3. механика цитат (бесплатно): ссылки [i] существуют, цитата -
     подстрока контекста;
  4. self-check (ещё один вызов LLM): судья подтверждает ответ контекстом;
     провал любого из рубежей 3-4 -> retry генерации с t=0.3, максимум раз.

Ответ, не прошедший проверки после retry, - вердикт refused_after_checks:
честный отказ вместо красиво оформленной выдумки.
"""

from __future__ import annotations

import re
import time
from typing import Awaitable, Callable

import structlog

from .logs import question_fingerprint
from .sanitizer import sanitize_question

log = structlog.get_logger()

REFUSAL_TEXT = "В базе знаний нет ответа на этот вопрос"

# промпт v3 из урока 3.3: решение об отказе отделено от стиля ответа
SYSTEM_RAG = (
    "Ты — ассистент по внутренней базе знаний компании «Векторика».\n"
    "Правила:\n"
    "1. Отвечай ТОЛЬКО на основе фрагментов из блока <context>.\n"
    "2. Сначала реши: есть ли в контексте ПРЯМОЙ ответ именно на этот вопрос "
    "(а не на похожий). Если нет — ответь ровно одной фразой: "
    "«В базе знаний нет ответа на этот вопрос» и больше ничего не пиши.\n"
    "3. Если ответ есть — дай его одним полным предложением, включая все "
    "условия, сроки и цифры из контекста по теме вопроса.\n"
    "4. В конце укажи использованные источники в виде [номер]. По-русски."
)

# рубеж 2 (урок 3.4): просим дословную цитату - её проверит механика ниже.
# Оговорка «только когда ответ есть» - не перестраховка: без неё qwen2.5:3b
# в живом прогоне выдал отказ И СЛЕДОМ идеальную цитату с ответом - правило
# цитаты перевесило правило «при отказе больше ничего не пиши»
CITE_ADDON = (
    "\n5. Правило только для случая, когда ответ ЕСТЬ: перед списком "
    "источников приведи ОДНУ дословную цитату из контекста (3-15 слов), "
    "на которой основан ответ, в формате:\n"
    "<quote>точная цитата из фрагмента</quote>\n"
    "При отказе по правилу 2 цитата и источники НЕ нужны."
)

# рубеж 4 (урок 3.4): строгий судья из урока 3.3, калибровка 8/8
VERIFY_SYSTEM = (
    "Ты проверяешь факты. Дан фрагмент документа и утверждение. Ответь одним "
    "словом: «да» — если утверждение прямо подтверждается фрагментом, «нет» — "
    "если не подтверждается или противоречит."
)

# агентный ход (урок 3.5): мусорная выдача лечится другими словами, а не
# другим порогом; подсказка про синонимы - как в description инструмента
REFORMULATE_SYSTEM = (
    "Ты помогаешь поиску по базе знаний компании. Переформулируй вопрос "
    "пользователя для поиска: замени разговорные обороты официальными "
    "терминами и синонимами (например, «упал прод» → «инцидент в production, "
    "дежурный»). Ответь ТОЛЬКО переформулированным запросом, без кавычек "
    "и пояснений."
)

REFUSAL_MARKERS = (
    "нет ответа",
    "нет информации",
    "не нашлось",
    "не могу ответить",
    "нет данных",
)

SearchFn = Callable[[str, int], Awaitable[list[dict]]]
GenerateFn = Callable[..., Awaitable[str]]      # generate(messages, **options)


def build_prompt(question: str, chunks: list[dict], cite: bool = False) -> list[dict]:
    """Нумерованный контекст в разделителях + вопрос (урок 3.1).

    cite=True добавляет требование дословной цитаты (рубеж 2, урок 3.4)."""
    context = "\n\n".join(
        f"[{i}] {chunk['title']}:\n{chunk['text']}"
        for i, chunk in enumerate(chunks, start=1)
    )
    return [
        {"role": "system", "content": SYSTEM_RAG + (CITE_ADDON if cite else "")},
        {
            "role": "user",
            "content": f"<context>\n{context}\n</context>\n\nВопрос: {question}",
        },
    ]


def build_verify_messages(answer: str, found: list[dict]) -> list[dict]:
    """Промпт self-check: фрагмент ПЕРЕД утверждением (порядок важен, урок 3.3)."""
    context = "\n".join(c["text"] for c in found)
    claim = re.sub(r"<quote>.+?</quote>|\[\d+\]", "", answer, flags=re.DOTALL)
    return [
        {"role": "system", "content": VERIFY_SYSTEM},
        {"role": "user",
         "content": f"Фрагмент:\n{context}\n\nУтверждение: {claim.strip()}"},
    ]


def build_reformulate_messages(question: str) -> list[dict]:
    return [{"role": "system", "content": REFORMULATE_SYSTEM},
            {"role": "user", "content": question}]


def is_refusal(answer: str) -> bool:
    """Детектор отказа по паттернам (урок 3.4, случай parking)."""
    lowered = answer.lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def citations_valid(answer: str, n_chunks: int) -> bool:
    """Рубеж 3, проверка 1: ссылки [i] существуют и есть хотя бы одна."""
    refs = [int(x) for x in re.findall(r"\[(\d+)\]", answer)]
    if not refs:
        return False
    return all(1 <= ref <= n_chunks for ref in refs)


def _normalize(text: str) -> str:
    """Сравниваем без регистра, пунктуации и лишних пробелов."""
    return re.sub(r"[^\wа-яё ]", "", text.lower(), flags=re.IGNORECASE).strip()


def quote_grounded(answer: str, found: list[dict]) -> bool | None:
    """Рубеж 3, проверка 2: цитата из <quote> дословно есть в контексте.

    None - модель не дала цитату: мягкий сигнал, решение за self-check."""
    match = re.search(r"<quote>(.+?)</quote>", answer, re.DOTALL)
    if match is None:
        return None
    quote = _normalize(match.group(1))
    context = _normalize(" ".join(c["text"] for c in found))
    return quote in context


def strip_service_tags(answer: str) -> str:
    """Убирает <quote>...</quote> из ответа: цитата - для проверки, не для людей."""
    clean = re.sub(r"<quote>.+?</quote>", "", answer, flags=re.DOTALL)
    return re.sub(r" {2,}", " ", clean).strip()


def _refusal_answer(chunks: list[dict]) -> str:
    """Канонический отказ + подсказка ближайших тем (урок 3.4)."""
    topics: list[str] = []
    for chunk in chunks:
        title = chunk.get("title", "")
        if title and title not in topics:
            topics.append(title)
    hint = f" Ближайшие темы в базе: {', '.join(topics[:3])}." if topics else ""
    return f"{REFUSAL_TEXT}.{hint}"


class RagService:
    """Конвейер: санитизация -> поиск (+агентный retry) -> порог ->
    генерация -> механика цитат -> self-check (+retry) -> вердикт."""

    def __init__(
        self,
        search: SearchFn,
        generate: GenerateFn,
        score_threshold: float = 0.10,
        guardrails: bool = True,
        agent_mode: bool = True,
        max_retries: int = 1,
    ):
        self.search = search
        self.generate = generate
        self.score_threshold = score_threshold
        self.guardrails = guardrails
        self.agent_mode = agent_mode
        self.max_retries = max_retries

    def _below_threshold(self, found: list[dict]) -> bool:
        return not found or found[0]["score"] < self.score_threshold

    @staticmethod
    def _result(answer: str, verdict: str, sanitizer: str, *,
                sources: list[dict] | None = None, checks: dict | None = None,
                search_ms: float = 0.0, llm_ms: float = 0.0,
                llm_calls: int = 0) -> dict:
        return {"answer": answer, "sources": sources or [], "verdict": verdict,
                "sanitizer": sanitizer, "checks": checks or {},
                "search_ms": round(search_ms, 1), "llm_ms": round(llm_ms, 1),
                "llm_calls": llm_calls}

    async def ask(self, raw_question: str, top_k: int = 3) -> dict:
        checks: dict = {}
        llm_ms = 0.0
        llm_calls = 0

        async def call_llm(messages: list[dict], **options) -> str:
            nonlocal llm_ms, llm_calls
            t = time.perf_counter()
            text = await self.generate(messages, **options)
            llm_ms += (time.perf_counter() - t) * 1000
            llm_calls += 1
            return text

        # --- рубеж 0: санитизация (урок 3.7); LLM ещё не звали ----------------
        checked = sanitize_question(raw_question)
        if checked["verdict"] == "rejected":
            log.warning("sanitizer_rejected", reason=checked["reason"])
            return self._result("Вопрос не прошёл проверку.",
                                "rejected_input", checked["verdict"])
        if checked["verdict"] == "suspicious":
            log.warning("sanitizer_suspicious", reason=checked["reason"],
                        patterns=checked.get("patterns", []))
        question = checked["question"]

        # --- поиск + порог релевантности (уроки 3.2, 3.4) ---------------------
        t0 = time.perf_counter()
        found = await self.search(question, top_k)
        search_ms = (time.perf_counter() - t0) * 1000
        log.info("retrieval_done", attempt=1,
                 top_score=round(found[0]["score"], 3) if found else 0.0,
                 doc_ids=[c.get("doc_id") for c in found],
                 search_ms=round(search_ms, 1))

        if self._below_threshold(found):
            if self.agent_mode:
                # --- агентный ход (урок 3.5): поискать снова другими словами --
                # сам текст переформулировки в лог не пишем (производная от
                # вопроса - те же ПДн), только отпечаток
                alt = (await call_llm(build_reformulate_messages(question),
                                      num_predict=60)).strip().strip('"«»')
                checks["reformulated"] = alt
                log.info("query_reformulated", **question_fingerprint(alt))
                t1 = time.perf_counter()
                found = await self.search(alt, top_k)
                search_ms += (time.perf_counter() - t1) * 1000
                log.info("retrieval_done", attempt=2,
                         top_score=round(found[0]["score"], 3) if found else 0.0,
                         doc_ids=[c.get("doc_id") for c in found],
                         search_ms=round(search_ms, 1))
            if self._below_threshold(found):
                log.info("threshold_refused",
                         top_score=round(found[0]["score"], 3) if found else 0.0)
                return self._result(f"{REFUSAL_TEXT}.", "refused_by_threshold",
                                    checked["verdict"], checks=checks,
                                    search_ms=search_ms, llm_ms=llm_ms,
                                    llm_calls=llm_calls)

        # --- генерация + рубежи 2-4 (уроки 3.1-3.4) ---------------------------
        messages = build_prompt(question, found, cite=self.guardrails)
        for attempt in range(1 + (self.max_retries if self.guardrails else 0)):
            # retry - с лёгкой температурой: при t=0 вторая попытка почти
            # наверняка дословно повторит первую (урок 3.4)
            answer = await call_llm(messages,
                                    temperature=0.0 if attempt == 0 else 0.3)

            if is_refusal(answer):                    # модель отказалась сама
                checks["model_refused"] = True
                return self._result(_refusal_answer(found), "refused_by_model",
                                    checked["verdict"], checks=checks,
                                    search_ms=search_ms, llm_ms=llm_ms,
                                    llm_calls=llm_calls)

            if not self.guardrails:                   # доверяем модели как есть
                return self._result(answer, "ok", checked["verdict"],
                                    sources=found, checks=checks,
                                    search_ms=search_ms, llm_ms=llm_ms,
                                    llm_calls=llm_calls)

            # --- рубеж 3: механика цитат (бесплатно) --------------------------
            checks["citations"] = citations_valid(answer, len(found))
            checks["quote"] = quote_grounded(answer, found)
            if not checks["citations"] or checks["quote"] is False:
                checks[f"attempt_{attempt + 1}"] = "citation_fail"
                log.warning("checks_failed", stage="citations",
                            attempt=attempt + 1)
                continue                              # retry генерации

            # --- рубеж 4: self-check (ещё один вызов LLM, урок 3.4) -----------
            raw = await call_llm(build_verify_messages(answer, found),
                                 num_predict=10)
            checks["self_check"] = raw.strip().lower().startswith("да")
            if checks["self_check"]:
                return self._result(strip_service_tags(answer), "ok",
                                    checked["verdict"], sources=found,
                                    checks=checks, search_ms=search_ms,
                                    llm_ms=llm_ms, llm_calls=llm_calls)
            checks[f"attempt_{attempt + 1}"] = "self_check_fail"
            log.warning("checks_failed", stage="self_check", attempt=attempt + 1)

        # все попытки провалили проверки - честный отказ (урок 3.4)
        return self._result(_refusal_answer(found), "refused_after_checks",
                            checked["verdict"], checks=checks,
                            search_ms=search_ms, llm_ms=llm_ms,
                            llm_calls=llm_calls)
