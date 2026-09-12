#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""kb-search — поиск по базе знаний «Векторики».

Финальный проект модуля 2 курса «LLM для Python-разработчиков».
Здесь работает весь арсенал модуля:

  - чанкинг по абзацам + контекстные заголовки          (уроки 2.5, 2.6)
  - метаданные-паспорт каждого чанка                    (урок 2.5)
  - два бэкенда за одним интерфейсом:
      lexical  — TF-IDF, работает сразу, без моделей    (уроки 2.5, 2.6)
      semantic — e5 + Qdrant, включается автоматически,
                 когда модель в кэше и сервер запущен   (уроки 2.1, 2.4)
  - pre-filter по теме и диверсификация выдачи          (уроки 2.2, 2.4, 2.5)
  - eval как встроенная команда                         (урок 1.7 навсегда)
  - parent-document: окно вокруг находки или родитель   (урок 2.6)
  - semantic chunking: границы по смысловой кривой      (урок 2.6)
  - вторая ступень: cross-encoder или LLM-судья         (урок 2.6 + модуль 1)
  - снапшоты Qdrant как команда backup                  (урок 2.4)

Примеры:
    python project/kb_search.py health
    python project/kb_search.py index
    python project/kb_search.py index --chunking semantic
    python project/kb_search.py search "Сколько доплачивают за болезнь?"
    python project/kb_search.py search "Куда писать про сломанный ноутбук?" --topic it
    python project/kb_search.py search "Как оформить больничный и отпуск?" --diverse
    python project/kb_search.py search "Сколько доплачивают за болезнь?" --parent
    python project/kb_search.py search "Упал прод, что делать?" --rerank ce
    python project/kb_search.py eval
    python project/kb_search.py compare
    python project/kb_search.py backup
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

# --- пути: утилита живёт в project/, модульные utils и data — уровнем выше ---
MODULE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MODULE_DIR))
from utils import (  # noqa: E402
    chunk_fixed,
    chunk_paragraphs,
    load_documents,
    split_sentences,
)

# --- конфигурация из окружения (привычка из урока 1.4) ------------------------
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
EMB_MODEL_NAME = os.getenv("KB_EMB_MODEL", "intfloat/multilingual-e5-small")
COLLECTION = os.getenv("KB_COLLECTION", "vectorika_kb")
CE_MODEL_NAME = os.getenv("KB_CE_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
OLLAMA_URL = os.getenv("KB_OLLAMA_URL", "http://127.0.0.1:11434")
LLM_MODEL = os.getenv("KB_LLM_MODEL", "qwen2.5:3b")
DATA_DIR = MODULE_DIR / "data"
LEXICAL_INDEX_PATH = Path(__file__).resolve().parent / "index_lexical.json"

# --- UTF-8 для Windows-консоли и пайпов (как в llm-cli модуля 1) --------------
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


# --- модели строго из кэша: утилита никогда не ходит в Hub за спиной -----------
_E5_MODEL = None


def hf_offline():
    """Выключает походы в Hub целиком, а заодно прогресс-бары и отчёты
    загрузки весов: CLI они не нужны. Без оффлайн-режима загрузка модели
    из кэша может зависнуть на минуты, если Hub недоступен по сети."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")


def load_e5():
    global _E5_MODEL
    if _E5_MODEL is None:
        hf_offline()
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(EMB_MODEL_NAME, local_files_only=True)
        except Exception:
            raise RuntimeError(
                f"Модель {EMB_MODEL_NAME} не в кэше — semantic-бэкенду и "
                "semantic-чанкингу нужно её скачать (урок 2.1)"
            ) from None
        from sentence_transformers import SentenceTransformer

        _E5_MODEL = SentenceTransformer(EMB_MODEL_NAME)
    return _E5_MODEL


# --- конвейер индексации (уроки 2.5-2.6) --------------------------------------
def split_semantic(text: str, max_chars: int = 500) -> list[str]:
    """Нарезка по «долинам» смысловой кривой (урок 2.6, акт 2).

    Каждое предложение эмбеддится один раз; режем там, где окно «до» и окно
    «после» наименее похожи (нижние 10% кривой), потом склеиваем сегменты
    до max_chars — так же, как chunk_paragraphs склеивает абзацы.
    """
    import numpy as np

    model = load_e5()  # e5 нужна уже на индексации, даже если искать будет lexical
    sentences = split_sentences(text)
    if len(sentences) < 4:  # резать нечего
        return chunk_paragraphs(text, max_chars=max_chars)

    sent_vecs = model.encode(
        [f"query: {s}" for s in sentences], normalize_embeddings=True
    )

    def window_vec(indices) -> np.ndarray:
        v = sent_vecs[list(indices)].mean(axis=0)
        return v / np.linalg.norm(v)

    W = 2
    curve = [
        float(
            window_vec(range(max(0, i - W), i))
            @ window_vec(range(i, min(len(sentences), i + W)))
        )
        for i in range(1, len(sentences))
    ]
    threshold = float(np.percentile(curve, 10))  # нижние 10% - кандидаты в границы

    segments, seg = [], [sentences[0]]
    for i, sim in enumerate(curve, start=1):
        if sim <= threshold:
            segments.append(" ".join(seg))
            seg = []
        seg.append(sentences[i])
    segments.append(" ".join(seg))

    # сегменты дальше ведут себя как абзацы: короткие склеиваем, длинные дорезаем
    merged: list[str] = []
    buf = ""
    for segment in segments:
        if len(segment) > max_chars:
            if buf:
                merged.append(buf)
                buf = ""
            merged.extend(chunk_fixed(segment, chunk_size=max_chars, overlap=100))
            continue
        candidate = f"{buf} {segment}" if buf else segment
        if len(candidate) <= max_chars:
            buf = candidate
        else:
            merged.append(buf)
            buf = segment
    if buf:
        merged.append(buf)
    return merged


def build_chunks(max_chars: int = 500, strategy: str = "paragraphs") -> list[dict]:
    """Документы -> чанки с паспортами и контекстными заголовками."""
    splitter = split_semantic if strategy == "semantic" else chunk_paragraphs
    chunks: list[dict] = []
    for doc in load_documents(DATA_DIR):
        pieces = splitter(doc.text, max_chars=max_chars)
        for i, piece in enumerate(pieces):
            # индексируем обогащённый текст, показываем оригинал (урок 2.6);
            # чанку 0 заголовок не нужен - он и есть начало документа
            indexed = piece if i == 0 else f"{doc.title}. {piece}"
            chunks.append(
                {
                    "chunk_id": f"{doc.doc_id}_{i}",
                    "doc_id": doc.doc_id,
                    "topic": doc.topic,
                    "title": doc.title,
                    "chunk_index": i,
                    "text": piece,
                    "indexed_text": indexed,
                }
            )
    return chunks


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


# --- бэкенд 1: лексический (TF-IDF из уроков 2.5-2.6) -------------------------
class LexicalBackend:
    """Работает сразу: без моделей, без сервера. Индекс — JSON-файл рядом."""

    name = "lexical"

    def index(self, chunks: list[dict]) -> int:
        LEXICAL_INDEX_PATH.write_text(
            json.dumps(chunks, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        return len(chunks)

    def _load(self):
        if not LEXICAL_INDEX_PATH.exists():
            sys.exit(
                "Индекс не построен — сначала выполните: python kb_search.py index"
            )
        chunks = json.loads(LEXICAL_INDEX_PATH.read_text(encoding="utf-8"))
        token_lists = [tokenize(c["indexed_text"]) for c in chunks]
        df: dict[str, int] = {}
        for tokens in token_lists:
            for word in set(tokens):
                df[word] = df.get(word, 0) + 1
        idf = {w: math.log(len(chunks) / n) for w, n in df.items()}
        return chunks, token_lists, idf

    def search(self, query: str, top_k: int = 10, topic: str | None = None):
        chunks, token_lists, idf = self._load()
        query_words = set(tokenize(query))
        results = []
        for chunk, tokens in zip(chunks, token_lists):
            if topic and chunk["topic"] != topic:  # pre-filter, а не post (урок 2.2)
                continue
            score = sum(
                (tokens.count(w) / len(tokens)) * idf.get(w, 0.0) for w in query_words
            )
            results.append((score, chunk))
        return sorted(results, key=lambda pair: -pair[0])[:top_k]


# --- бэкенд 2: семантический (e5 + Qdrant, уроки 2.1 и 2.4) --------------------
class SemanticBackend:
    """Основной стек модуля. Импорты ленивые: без модели бэкенд просто не создастся."""

    name = "semantic"

    def __init__(self):
        from qdrant_client import QdrantClient

        self.model = load_e5()  # только из кэша
        self.client = QdrantClient(url=QDRANT_URL, timeout=60)
        self.client.get_collections()  # проверка, что сервер отвечает

    def _embed(self, texts: list[str], prefix: str):
        # префиксы e5 обязательны - урок 2.1
        return self.model.encode(
            [f"{prefix}: {t}" for t in texts], normalize_embeddings=True
        )

    def index(self, chunks: list[dict]) -> int:
        import uuid

        from qdrant_client.models import Distance, PointStruct, VectorParams

        # пересоздаём: другая стратегия чанкинга даёт другое число чанков,
        # и uuid5-upsert оставил бы точки-сироты от старой нарезки
        if self.client.collection_exists(COLLECTION):
            self.client.delete_collection(COLLECTION)
        self.client.create_collection(
            COLLECTION,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
        self.client.create_payload_index(
            COLLECTION, field_name="topic", field_schema="keyword"
        )

        vectors = self._embed([c["indexed_text"] for c in chunks], "passage")
        self.client.upsert(
            COLLECTION,
            points=[
                PointStruct(
                    # id только int/UUID: детерминированный uuid5 из chunk_id (урок 2.4)
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk["chunk_id"])),
                    vector=vectors[i].tolist(),
                    payload=chunk,
                )
                for i, chunk in enumerate(chunks)
            ],
        )
        return len(chunks)

    def search(self, query: str, top_k: int = 10, topic: str | None = None):
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        query_filter = None
        if topic:
            query_filter = Filter(
                must=[FieldCondition(key="topic", match=MatchValue(value=topic))]
            )
        q_vec = self._embed([query], "query")[0]
        response = self.client.query_points(
            COLLECTION, query=q_vec.tolist(), limit=top_k, query_filter=query_filter
        )
        return [(p.score, p.payload) for p in response.points]


def pick_backend(name: str):
    """auto: semantic, если модель в кэше и Qdrant отвечает; иначе lexical."""
    if name == "lexical":
        return LexicalBackend()
    if name == "semantic":
        return SemanticBackend()  # упадёт с понятной ошибкой, если чего-то нет
    try:
        return SemanticBackend()
    except Exception:
        print(
            "[semantic недоступен — работаю на lexical; подробности: health]",
            file=sys.stderr,
        )
        return LexicalBackend()


# --- reranking: диверсификация из урока 2.5 ------------------------------------
def rerank_diverse(results, max_per_doc: int = 1):
    picked, per_doc = [], {}
    for score, chunk in results:
        owner = chunk["doc_id"]
        if per_doc.get(owner, 0) >= max_per_doc:
            continue
        picked.append((score, chunk))
        per_doc[owner] = per_doc.get(owner, 0) + 1
    return picked


# --- parent-document: отдаём больше контекста, чем нашли (урок 2.6) -------------
def parent_window(chunk: dict, parent_text: str, radius: int = 300) -> str:
    """Кусок родителя в +-radius символов вокруг найденного чанка, по словам."""
    start = parent_text.find(chunk["text"][:50])  # где чанк лежит в родителе
    if start == -1:
        return chunk["text"]  # не нашли (чанк склеен из предложений) - сам чанк
    left = max(0, start - radius)
    right = min(len(parent_text), start + len(chunk["text"]) + radius)
    # не режем слова по краям
    if left > 0:
        left = parent_text.find(" ", left) + 1
    if right < len(parent_text):
        right = parent_text.rfind(" ", left, right)
    return (
        ("..." if left > 0 else "")
        + parent_text[left:right].strip()
        + ("..." if right < len(parent_text) else "")
    )


# --- вторая ступень: cross-encoder или LLM-судья (урок 2.6 + модуль 1) ----------
class RerankUnavailable(Exception):
    """Вторая ступень не готова: модель не в кэше или Ollama молчит."""


def rerank_cross_encoder(question: str, results):
    hf_offline()
    try:
        from huggingface_hub import snapshot_download
        from sentence_transformers import CrossEncoder

        snapshot_download(CE_MODEL_NAME, local_files_only=True)  # только из кэша
        ce_model = CrossEncoder(CE_MODEL_NAME)
    except Exception:
        raise RerankUnavailable(
            f"cross-encoder не в кэше — скачайте один раз (~500 МБ): python -c "
            f'"from sentence_transformers import CrossEncoder; '
            f"CrossEncoder('{CE_MODEL_NAME}')\""
        ) from None
    # cross-encoder читает пары (вопрос, чанк) целиком - счёт получает каждая пара
    scores = ce_model.predict([(question, chunk["text"]) for _, chunk in results])
    rescored = [(float(s), chunk) for s, (_, chunk) in zip(scores, results)]
    return sorted(rescored, key=lambda pair: -pair[0])


def build_judge_prompt(question: str, fragment: str) -> str:
    """Промпт бинарного LLM-судьи — два живых урока в одной функции.

    Батч-промпт со шкалой 0-10 (решение 3 урока 2.6) с 3B-судьёй разваливается
    (в нашем прогоне: 10 больничному на вопрос про упавший прод): судья надёжен
    только с ответом да/нет (урок 3.3). И вопрос должен стоять ПОСЛЕ фрагмента,
    ближе к месту решения (урок 1.2): с вопросом в начале тот же судья отвечал
    «нет» даже дословным совпадениям.
    """
    return f"""Фрагмент документа:
<fragment>
{fragment}
</fragment>

Вопрос: {question}

Есть ли во фрагменте ответ на вопрос? Ответь одним словом: да или нет."""


def rerank_llm(question: str, results):
    import httpx

    rescored = []
    for _, chunk in results:  # по кандидату на вызов - как LLMReranker в уроке 4.4
        payload = {
            "model": LLM_MODEL,
            "prompt": build_judge_prompt(question, chunk["text"]),
            "stream": False,
            "options": {"temperature": 0, "num_predict": 5},
        }
        try:
            response = httpx.post(
                f"{OLLAMA_URL}/api/generate", json=payload, timeout=180
            )
        except httpx.HTTPError:
            raise RerankUnavailable(
                f"Ollama не отвечает на {OLLAMA_URL} — запустите её (модуль 1) "
                "или укажите KB_OLLAMA_URL"
            ) from None
        if response.status_code == 404:
            raise RerankUnavailable(
                f"модель {LLM_MODEL} не скачана: ollama pull {LLM_MODEL}"
            )
        response.raise_for_status()
        answer = response.json()["response"].strip().lower()
        rescored.append((1.0 if answer.startswith("да") else 0.0, chunk))
    # sorted стабильна: внутри групп «да» и «нет» порядок первой ступени сохраняется
    return sorted(rescored, key=lambda pair: -pair[0])


# --- eval: семь вопросов модуля (урок 1.7 навсегда) -----------------------------
EVAL_QUESTIONS = [
    ("Сколько дней отпуска положено в году?", "otpusk"),
    ("Как оформить больничный лист?", "bolnichny"),
    ("VPN подключён, но внутренние сайты не открываются", "vpn"),
    ("Кто должен посмотреть мой код перед мержем?", "code-review"),
    ("Упал прод, что делать в первую очередь?", "incidents"),
    ("Сколько компания доплачивает в первые дни болезни?", "bolnichny"),
    ("За сколько минут дежурный должен отреагировать ночью?", "incidents"),
]


# --- команды -------------------------------------------------------------------
def cmd_health(args: argparse.Namespace) -> int:
    docs = load_documents(DATA_DIR)
    print(f"[ OK ] данные: {len(docs)} документов в {DATA_DIR.name}/")

    try:
        import httpx

        version = httpx.get(QDRANT_URL, timeout=3).json()["version"]
        print(f"[ OK ] Qdrant {version} на {QDRANT_URL}")
    except Exception:
        print(
            f"[WARN] Qdrant не отвечает на {QDRANT_URL} — semantic-бэкенду нужен"
            " запущенный сервер (docker compose up -d)"
        )

    try:
        from huggingface_hub import snapshot_download

        snapshot_download(EMB_MODEL_NAME, local_files_only=True)
        print(f"[ OK ] модель {EMB_MODEL_NAME} в кэше")
    except Exception:
        print(
            f"[WARN] модель {EMB_MODEL_NAME} не скачана — доступен только"
            " lexical-бэкенд (установка: урок 2.1)"
        )

    if LEXICAL_INDEX_PATH.exists():
        n = len(json.loads(LEXICAL_INDEX_PATH.read_text(encoding="utf-8")))
        print(f"[ OK ] lexical-индекс построен: {n} чанков")
    else:
        print("[INFO] lexical-индекс не построен — python kb_search.py index")

    # опциональная вторая ступень (search --rerank)
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(CE_MODEL_NAME, local_files_only=True)
        print(f"[ OK ] cross-encoder {CE_MODEL_NAME} в кэше — rerank ce доступен")
    except Exception:
        print("[INFO] cross-encoder не скачан — rerank ce недоступен (урок 2.6)")
    try:
        import httpx

        version = httpx.get(f"{OLLAMA_URL}/api/version", timeout=3).json()["version"]
        print(f"[ OK ] Ollama {version} на {OLLAMA_URL} — rerank llm доступен")
    except Exception:
        print(
            f"[INFO] Ollama не отвечает на {OLLAMA_URL} — rerank llm недоступен"
            " (модуль 1)"
        )
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    backend = pick_backend(args.backend)
    try:
        chunks = build_chunks(max_chars=args.max_chars, strategy=args.chunking)
    except RuntimeError as exc:  # semantic-чанкинг без e5 в кэше
        sys.exit(str(exc))
    n = backend.index(chunks)
    docs_count = len({c["doc_id"] for c in chunks})
    print(
        f"[{backend.name}] проиндексировано: {docs_count} документов -> {n} чанков"
        f" (чанкинг: {args.chunking})"
    )
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    backend = pick_backend(args.backend)
    results = backend.search(args.query, top_k=10, topic=args.topic)
    if args.diverse:
        results = rerank_diverse(results)  # не более 1 чанка с документа

    if not results or results[0][0] <= 0:
        print("Ничего не нашлось. Попробуйте переформулировать запрос.")
        return 1

    if args.rerank:
        reranker = {"ce": rerank_cross_encoder, "llm": rerank_llm}[args.rerank]
        try:
            results = reranker(args.query, results)
            print(
                "Вторая ступень: "
                + {
                    "ce": "cross-encoder (счёт — логит, бывает отрицательным)",
                    "llm": f"LLM-судья {LLM_MODEL} (да=1/нет=0, ничьи — порядком первой ступени)",
                }[args.rerank]
                + "\n"
            )
        except RerankUnavailable as exc:
            print(
                f"[rerank {args.rerank} недоступен: {exc} — показываю выдачу"
                " первой ступени]",
                file=sys.stderr,
            )
    results = results[: args.top]

    parents = {}
    if args.parent:
        parents = {doc.doc_id: doc for doc in load_documents(DATA_DIR)}

    for rank, (score, chunk) in enumerate(results, start=1):
        print(
            f"{rank}. [{score:.3f}] {chunk['title']}  "
            f"({chunk['chunk_id']}, тема: {chunk['topic']})"
        )
        text = chunk["text"]
        limit = 200
        if args.parent:
            parent_text = parents[chunk["doc_id"]].text
            if args.parent == "full":
                expanded, label = parent_text, "родитель целиком"
            else:
                expanded, label = parent_window(chunk, parent_text), "окно +-300"
            print(f"   чанк {len(text)} симв. -> {label}: {len(expanded)} симв.")
            text, limit = expanded, 400
        print(
            f"   {text[:limit].replace(chr(10), ' ')}"
            f"{'...' if len(text) > limit else ''}\n"
        )
    return 0


def run_eval(backend) -> dict:
    """Прогоняет встроенный eval-набор; отметки: + top-1, ~ top-3, x мимо."""
    started = time.perf_counter()
    top1 = top3 = total_chars = 0
    marks: list[str] = []
    for question, expected in EVAL_QUESTIONS:
        results = backend.search(question, top_k=10)
        found_docs: list[str] = []
        for _, chunk in results:
            if chunk["doc_id"] not in found_docs:
                found_docs.append(chunk["doc_id"])
        hit1 = bool(found_docs) and found_docs[0] == expected
        hit3 = expected in found_docs[:3]
        top1 += hit1
        top3 += hit3
        total_chars += len(results[0][1]["text"]) if results else 0
        marks.append("+" if hit1 else ("~" if hit3 else "x"))
    return {
        "top1": top1,
        "top3": top3,
        "avg_chars": total_chars // len(EVAL_QUESTIONS),
        "seconds": time.perf_counter() - started,
        "marks": marks,
    }


def cmd_eval(args: argparse.Namespace) -> int:
    backend = pick_backend(args.backend)
    stats = run_eval(backend)
    for mark, (question, _) in zip(stats["marks"], EVAL_QUESTIONS):
        print(f"[{mark}] {question}")

    total = len(EVAL_QUESTIONS)
    print(
        f"\n[{backend.name}] top-1: {stats['top1']}/{total}   "
        f"top-3: {stats['top3']}/{total}   "
        f"средняя находка: {stats['avg_chars']} симв."
    )
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Один eval — оба бэкенда: таблица вместо впечатлений."""
    columns: dict[str, dict] = {}
    for name, factory in (("lexical", LexicalBackend), ("semantic", SemanticBackend)):
        try:
            columns[name] = run_eval(factory())
        except SystemExit as exc:  # lexical без индекса выходит с подсказкой
            print(f"[{name} недоступен: {exc}]", file=sys.stderr)
        except Exception as exc:
            print(
                f"[{name} недоступен: {str(exc)[:100]} — подробности: health]",
                file=sys.stderr,
            )
    if not columns:
        sys.exit("Ни один бэкенд не готов — начните с: python kb_search.py health")

    names = list(columns)
    width = max(len(q) for q, _ in EVAL_QUESTIONS) + 2
    print(f"{'вопрос':<{width}}" + "".join(f"{n:>10}" for n in names))
    for i, (question, _) in enumerate(EVAL_QUESTIONS):
        row = "".join(f"{columns[n]['marks'][i]:>10}" for n in names)
        print(f"{question:<{width}}{row}")

    total = len(EVAL_QUESTIONS)
    print("-" * (width + 10 * len(names)))
    metric_rows = (
        ("top-1", lambda s: f"{s['top1']}/{total}"),
        ("top-3", lambda s: f"{s['top3']}/{total}"),
        ("средняя находка, симв.", lambda s: str(s["avg_chars"])),
        ("время eval, с", lambda s: f"{s['seconds']:.2f}"),
    )
    for label, fmt in metric_rows:
        print(f"{label:<{width}}" + "".join(f"{fmt(columns[n]):>10}" for n in names))
    if len(columns) < 2:
        print("\n(для настоящего сравнения нужны оба бэкенда)")
    return 0


def cmd_backup(args: argparse.Namespace) -> int:
    """Снапшот коллекции Qdrant через API (урок 2.4): wait=False + поллинг."""
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=QDRANT_URL, timeout=60)
        client.get_collections()
    except Exception:
        sys.exit(f"Qdrant не отвечает на {QDRANT_URL} — docker compose up -d")
    if not client.collection_exists(COLLECTION):
        sys.exit(
            f"Коллекции {COLLECTION} нет — сначала:"
            " python kb_search.py --backend semantic index"
        )

    known = {s.name for s in client.list_snapshots(COLLECTION)}
    # wait=False: сервер начинает работу и сразу отвечает, не держа соединение;
    # готовность поллим - на Windows под Docker это бывает небыстро (урок 2.4)
    client.create_snapshot(collection_name=COLLECTION, wait=False)
    print("Снапшот строится в фоне, опрашиваем готовность...")

    snapshot = None
    for attempt in range(60):  # ждём до двух минут
        time.sleep(2)
        fresh = [s for s in client.list_snapshots(COLLECTION) if s.name not in known]
        if fresh:
            snapshot = fresh[-1]
            break
        if attempt % 5 == 4:
            print(f"  ...ещё строится ({(attempt + 1) * 2} с)")
    if snapshot is None:
        sys.exit(
            "За 2 минуты не построился — проверьте позже: "
            f"GET {QDRANT_URL}/collections/{COLLECTION}/snapshots"
        )

    print(f"Готов: {snapshot.name} ({(snapshot.size or 0) / 2**20:.1f} МБ)")
    print(
        "Скачать (файл живёт внутри контейнера, том его не переживёт): "
        f"GET {QDRANT_URL}/collections/{COLLECTION}/snapshots/{snapshot.name}"
    )
    print(
        "Восстановить на другом сервере: "
        f"PUT {QDRANT_URL}/collections/{COLLECTION}/snapshots/upload"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="kb-search",
        description="Поиск по базе знаний «Векторики». Бэкенд выбирается "
        "автоматически: semantic (e5+Qdrant), если доступен, иначе lexical.",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "lexical", "semantic"],
        default="auto",
        help="принудительный выбор бэкенда",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("health", help="диагностика: данные, Qdrant, модель, индекс")
    p.set_defaults(func=cmd_health)

    p = sub.add_parser("index", help="построить индекс из data/")
    p.add_argument("--max-chars", type=int, default=500, help="размер чанка")
    p.add_argument(
        "--chunking",
        choices=["paragraphs", "semantic"],
        default="paragraphs",
        help="нарезка: по абзацам или по смысловой кривой (semantic требует e5)",
    )
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("search", help="искать по базе")
    p.add_argument("query")
    p.add_argument("--top", type=int, default=3)
    p.add_argument("--topic", help="pre-filter по теме: hr / it / dev / office")
    p.add_argument(
        "--diverse", action="store_true", help="не более одного чанка с документа"
    )
    p.add_argument(
        "--parent",
        nargs="?",
        const="window",
        choices=["window", "full"],
        help="отдавать контекст вокруг находки: окно +-300 символов"
        " (по умолчанию) или родительский документ целиком",
    )
    p.add_argument(
        "--rerank",
        choices=["ce", "llm"],
        help="вторая ступень: cross-encoder или LLM-судья через Ollama",
    )
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("eval", help="прогнать встроенный eval-набор")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("compare", help="один eval — оба бэкенда, таблица-сравнение")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("backup", help="снапшот коллекции Qdrant (нужен сервер)")
    p.set_defaults(func=cmd_backup)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
