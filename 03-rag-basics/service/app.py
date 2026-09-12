"""RAG-сервис «Векторики»: FastAPI + async Qdrant + Ollama.

Тот же конвейер, что в уроке 3.1 (retrieve -> build_prompt -> chat),
но в форме сервиса: индекс живёт в Qdrant, все IO-вызовы асинхронные,
наружу торчит HTTP-API.

Запуск из папки модуля (сначала: python service/index_kb.py):
    uvicorn service.app:app --port 8010

Переменные окружения:
    QDRANT_URL   http://localhost:6333
    OLLAMA_URL   http://localhost:11434
    LLM_MODEL    qwen2.5:3b
    EMB_BACKEND  tfidf | e5 (коллекция и эмбеддер выбираются согласованно)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from service.embedder import make_embedder  # noqa: E402

# именно 127.0.0.1, а не localhost: Ollama слушает только IPv4, а httpx сперва
# пробует IPv6 ::1 - на Windows это ~2с штрафа на каждое новое соединение
QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:3b")
EMB_BACKEND = os.getenv("EMB_BACKEND", "e5")
COLLECTION = f"vectorika_rag_{EMB_BACKEND}"

# промпт - дословно из урока 3.1: сервис не меняет механику RAG
SYSTEM_RAG = (
    "Ты — ассистент по внутренней базе знаний компании «Векторика».\n"
    "Правила:\n"
    "1. Отвечай ТОЛЬКО на основе фрагментов из блока <context>. "
    "Не используй никакие другие знания.\n"
    "2. Если в контексте нет ответа на вопрос — честно скажи: "
    "«В базе знаний нет ответа на этот вопрос».\n"
    "3. В конце ответа укажи использованные источники в виде [номер].\n"
    "4. Отвечай кратко и по делу, по-русски."
)


def build_prompt(question: str, found_chunks: list[dict]) -> list[dict]:
    context = "\n\n".join(
        f"[{i}] {chunk['title']}:\n{chunk['text']}"
        for i, chunk in enumerate(found_chunks, start=1)
    )
    user_message = f"<context>\n{context}\n</context>\n\nВопрос: {question}"
    return [
        {"role": "system", "content": SYSTEM_RAG},
        {"role": "user", "content": user_message},
    ]


# --- жизненный цикл: тяжёлые объекты создаются ОДИН раз ---------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    from qdrant_client import AsyncQdrantClient

    state_path = Path(__file__).parent / "tfidf_state.json"
    app.state.embedder = make_embedder(
        EMB_BACKEND,
        state_path if EMB_BACKEND == "tfidf" else None,
    )
    app.state.qdrant = AsyncQdrantClient(url=QDRANT_URL)
    app.state.ollama = httpx.AsyncClient(
        base_url=OLLAMA_URL,
        timeout=httpx.Timeout(120, connect=5),
    )
    # GPU одна - больше одной генерации разом не пускаем, остальные ждут в очереди
    app.state.gen_semaphore = asyncio.Semaphore(
        int(os.getenv("RAG_MAX_CONCURRENT_GEN", "1"))
    )
    yield
    await app.state.qdrant.close()
    await app.state.ollama.aclose()


app = FastAPI(title="Vectorika RAG", version="0.1.0", lifespan=lifespan)


# --- модели запросов/ответов: Pydantic валидирует за нас --------------------
class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    top_k: int = Field(default=3, ge=1, le=10)
    temperature: float = Field(default=0.2, ge=0.0, le=1.0)


class Source(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    topic: str
    text: str
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    search_ms: float
    queue_ms: float  # ожидание в очереди к GPU (семафор) - очередь видна клиенту!
    llm_ms: float


# --- внутренности ------------------------------------------------------------
async def retrieve(request: Request, question: str, top_k: int) -> list[Source]:
    """Поиск в Qdrant. embed_query у TF-IDF занимает микросекунды, поэтому
    зовём его прямо в event loop; тяжёлый e5 пришлось бы уносить в поток
    (anyio.to_thread.run_sync) - см. «частые ошибки» урока 3.2."""
    vector = request.app.state.embedder.embed_query(question)
    hits = await request.app.state.qdrant.query_points(
        COLLECTION,
        query=vector,
        limit=top_k,
    )
    return [Source(**hit.payload, score=hit.score) for hit in hits.points]


async def ollama_chat(
    request: Request, messages: list[dict], temperature: float
) -> str:
    response = await request.app.state.ollama.post(
        "/api/chat",
        json={
            "model": LLM_MODEL,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": 400},
        },
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


# --- эндпоинты ----------------------------------------------------------------
@app.get("/healthz")
async def healthz(request: Request) -> dict:
    """Готовность зависимостей. Обе проверки идут ПАРАЛЛЕЛЬНО - gather."""

    async def check_qdrant():
        info = await request.app.state.qdrant.count(COLLECTION)
        return info.count

    async def check_ollama():
        response = await request.app.state.ollama.get("/api/tags", timeout=3)
        return any(m["name"] == LLM_MODEL for m in response.json()["models"])

    results = await asyncio.gather(
        check_qdrant(),
        check_ollama(),
        return_exceptions=True,
    )
    points, model_ready = (r if not isinstance(r, Exception) else None for r in results)
    ok = points is not None and model_ready is True
    return {
        "status": "ok" if ok else "degraded",
        "collection": COLLECTION,
        "points": points,
        "model": LLM_MODEL if model_ready else None,
        "embedder": request.app.state.embedder.name,
    }


@app.post("/search")
async def search(request: Request, body: AskRequest) -> list[Source]:
    try:
        return await retrieve(request, body.question, body.top_k)
    except Exception as error:  # Qdrant недоступен / коллекции нет
        raise HTTPException(status_code=503, detail=f"поиск недоступен: {error}")


@app.post("/ask")
async def ask(request: Request, body: AskRequest) -> AskResponse:
    t0 = time.perf_counter()
    try:
        sources = await retrieve(request, body.question, body.top_k)
    except Exception as error:
        raise HTTPException(status_code=503, detail=f"поиск недоступен: {error}")
    t1 = time.perf_counter()

    messages = build_prompt(body.question, [s.model_dump() for s in sources])
    try:
        async with request.app.state.gen_semaphore:
            t_gen = time.perf_counter()  # семафор взят - очередь кончилась
            answer = await ollama_chat(request, messages, body.temperature)
    except httpx.HTTPError as error:
        raise HTTPException(status_code=503, detail=f"LLM недоступна: {error}")
    t2 = time.perf_counter()

    return AskResponse(
        answer=answer,
        sources=sources,
        search_ms=round((t1 - t0) * 1000, 1),
        queue_ms=round((t_gen - t1) * 1000, 1),
        llm_ms=round((t2 - t_gen) * 1000, 1),
    )


@app.post("/ask-blocking")
async def ask_blocking(request: Request, body: AskRequest) -> AskResponse:
    """АНТИПРИМЕР для демо урока: синхронный httpx.post внутри async def.

    Пока крутится этот запрос, event loop СТОИТ: /healthz не отвечает,
    другие клиенты ждут. Никогда так не делайте - ячейка с демо в уроке
    показывает масштаб бедствия в миллисекундах.
    """
    t0 = time.perf_counter()
    sources = await retrieve(request, body.question, body.top_k)
    t1 = time.perf_counter()

    messages = build_prompt(body.question, [s.model_dump() for s in sources])
    response = httpx.post(
        f"{OLLAMA_URL}/api/chat",
        json={  # <- БЛОКИРУЕТ LOOP
            "model": LLM_MODEL,
            "messages": messages,
            "stream": False,
            "options": {"temperature": body.temperature, "num_predict": 400},
        },
        timeout=120,
    )
    response.raise_for_status()
    t2 = time.perf_counter()

    return AskResponse(
        answer=response.json()["message"]["content"],
        sources=sources,
        search_ms=round((t1 - t0) * 1000, 1),
        queue_ms=0.0,  # второй грех: лезет к GPU мимо очереди-семафора
        llm_ms=round((t2 - t1) * 1000, 1),
    )


@app.post("/ask/stream")
async def ask_stream(request: Request, body: AskRequest) -> StreamingResponse:
    """Стриминг в формате SSE: сначала источники, потом токены, потом done.

    Механика та же, что в решении 3 урока 3.1, но токены пересылаются
    дальше клиенту по мере прихода из Ollama.
    """
    try:
        sources = await retrieve(request, body.question, body.top_k)
    except Exception as error:
        raise HTTPException(status_code=503, detail=f"поиск недоступен: {error}")
    messages = build_prompt(body.question, [s.model_dump() for s in sources])

    async def event_stream():
        head = {
            "sources": [
                {k: s.model_dump()[k] for k in ("chunk_id", "title", "score")}
                for s in sources
            ]
        }
        yield f"data: {json.dumps(head, ensure_ascii=False)}\n\n"
        async with request.app.state.gen_semaphore:
            async with request.app.state.ollama.stream(
                "POST",
                "/api/chat",
                json={
                    "model": LLM_MODEL,
                    "messages": messages,
                    "stream": True,
                    "options": {"temperature": body.temperature, "num_predict": 400},
                },
            ) as response:
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    piece = json.loads(line)
                    delta = piece.get("message", {}).get("content", "")
                    if delta:
                        yield (
                            "data: "
                            + json.dumps({"delta": delta}, ensure_ascii=False)
                            + "\n\n"
                        )
                    if piece.get("done"):
                        yield 'data: {"done": true}\n\n'

    return StreamingResponse(event_stream(), media_type="text/event-stream")
