# RAG-сервис «Векторики» (урок 3.2)

Прототип из урока 3.1, превращённый в сервис: индекс в Qdrant, асинхронные
вызовы, HTTP-API. Урок `02-qdrant-async.ipynb` разбирает эти файлы по частям
и гоняет сервис живьём.

## Файлы

| файл | что делает |
|---|---|
| `embedder.py` | эмбеддеры за одним интерфейсом: `TfidfEmbedder` (работает сразу, состояние в `tfidf_state.json`) и `E5Embedder` (нужна скачанная e5) |
| `index_kb.py` | батч-индексация: `data/` → чанки с заголовками → векторы → Qdrant |
| `app.py` | FastAPI: `/healthz`, `/search`, `/ask`, `/ask/stream` (SSE) и антипример `/ask-blocking` для демо блокировки event loop |

## Запуск (из папки модуля `03-rag-basics`)

```bash
python service/index_kb.py
```

```bash
uvicorn service.app:app --host 127.0.0.1 --port 8010
```

Проверка:

```bash
curl http://127.0.0.1:8010/healthz
```

```bash
curl -X POST http://127.0.0.1:8010/ask -H "Content-Type: application/json" -d "{\"question\": \"Сколько дней отпуска положено в году?\"}"
```

## Конфигурация (переменные окружения)

| переменная | по умолчанию |
|---|---|
| `QDRANT_URL` | `http://127.0.0.1:6333` |
| `OLLAMA_URL` | `http://127.0.0.1:11434` |
| `LLM_MODEL` | `qwen2.5:3b` |
| `EMB_BACKEND` | `tfidf` (или `e5`; коллекция выберется согласованно) |
| `RAG_MAX_CONCURRENT_GEN` | `1` — размер очереди-семафора к GPU |

Везде `127.0.0.1`, а не `localhost`: Ollama и uvicorn слушают только IPv4,
а httpx сперва пробует IPv6 `::1` — на Windows это ~2 секунды штрафа на каждое
новое соединение (подробности — в «частых ошибках» урока 3.2).

## Переезд на e5

```bash
python service/index_kb.py --backend e5
```

```bash
EMB_BACKEND=e5 uvicorn service.app:app --host 127.0.0.1 --port 8010
```

Коллекции `vectorika_rag_tfidf` и `vectorika_rag_e5` живут раздельно —
«одна коллекция — одна модель». Патч для тяжёлого `e5.encode` (в поток через
`anyio.to_thread`) — в решении 3 урока.
