# Docs Assistant — AI-ассистент по внутренней документации

Финальный проект курса «LLM для Python-разработчиков: от RAG до агентов».
FastAPI-сервис: семантический поиск по документации (`/api/search`) и генерация
ответов с цитированием источников (`/api/generate`), включая стриминг SSE.
Всё работает локально и бесплатно: Qdrant + Ollama + модели с открытыми весами.

## Архитектура

```
клиент ──HTTP──▶ FastAPI (app/)
                   │ /api/search       │ /api/generate
                   ▼                   ▼
             EmbeddingService     RagService ──▶ OllamaClient ──▶ Ollama (qwen2.5:3b)
                   │                   │
                   ▼                   ▼
             Qdrant (ANN)        HistoryRepository ──▶ SQLite
                   │
             RerankerService (cross-encoder, опционально)
```

Решения зафиксированы в [docs/adr/](docs/adr/): стек, эмбеддинги, reranker,
хранилище истории, контракт API.

## Быстрый старт (локально)

Нужны: Python 3.13, запущенный Qdrant (`docker compose up -d qdrant` или свой)
и Ollama с моделью `qwen3.5:9b`.

```bash
python -m venv .venv && .venv\Scripts\activate     # Windows
pip install -r requirements.txt

# 1. Проиндексировать документацию (data/docs -> Qdrant):
python scripts/index_docs.py

# 2. Запустить сервис:
uvicorn app.main:app --port 8000

# 3. Проверить:
curl http://127.0.0.1:8000/ready
```

Первый запуск скачивает модель эмбеддингов (~120 МБ) и, если включён reranker,
cross-encoder (~500 МБ). Выключить reranker: `APP_RERANKER_ENABLED=false`.

## Примеры запросов

```bash
# Поиск
curl -X POST http://127.0.0.1:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "как откатить деплой на проде?", "top_k": 3}'

# Генерация (JSON)
curl -X POST http://127.0.0.1:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"query": "что делать при коммите секрета в git?"}'

# Генерация (стриминг SSE)
curl -N -X POST http://127.0.0.1:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"query": "как объявляется SEV1?", "stream": true}'

# История
curl "http://127.0.0.1:8000/api/history?limit=5"
```

## Контракт API

Схемы (`app/schemas.py`, [ADR-0005](docs/adr/0005-api-contract.md)) — строгие:

- тело запроса с неизвестным полем (`topk`), числом строкой (`"top_k": "3"`),
  булевым строкой или запросом из пробелов — **422**, до вызова RAG;
- правила на несколько полей проверяются в схеме: `fetch_k >= top_k`,
  `fetch_k` только вместе с reranker;
- ответы содержат вычисляемые поля: `count` и `ranked_by` у поиска;
  `cited`, `dangling_citations`, `is_refusal` у генерации; `failed` у `/ready`;
- SSE-события типизированы (`SourcesEvent | TokenEvent | DoneEvent | ErrorEvent`,
  `parse_sse` для клиентов), формат на проводе прежний;
- все ошибки, кроме 422, — `{"detail": ..., "request_id": ...}`;
- `created_at` в истории — ISO 8601 в UTC.

Интерактивная документация со схемами и примерами: `http://127.0.0.1:8000/docs`.

## Тесты

```bash
pytest            # юнит + интеграционные, без моделей/Qdrant/Ollama, < 1 c
```

Тяжёлые зависимости подменяются фейками на границе `app.state`
(tests/conftest.py) — тесты проверяют и вызовы, и НЕ-вызовы.

## Конфигурация

Все настройки — переменные окружения с префиксом `APP_` и/или файл `.env`
(см. [.env.example](.env.example) и `app/config.py`). Ключевые:

| переменная | по умолчанию | зачем |
|---|---|---|
| `APP_QDRANT_URL` | `http://127.0.0.1:6333` | адрес Qdrant |
| `APP_OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | адрес Ollama |
| `APP_LLM_MODEL` | `qwen2.5:3b` | модель генерации |
| `APP_LLM_THINK` | `false` | рассуждения thinking-моделей (qwen3.5+) перед ответом; для RAG выключены |
| `APP_LLM_NUM_CTX` | `8192` | контекст на один запрос (system + чанки + вопрос + ответ); дефолт сервера 4096 для RAG впритык |
| `APP_RERANKER_ENABLED` | `true` | вторая ступень поиска |
| `APP_SEARCH_TOP_K` / `APP_SEARCH_FETCH_K` | 5 / 20 | сколько отдать / сколько кандидатов до reranker |

## Наблюдаемость

`/health` — liveness (без внешних проверок), `/ready` — readiness (Qdrant +
Ollama, при деградации 503 со списком причин), `/metrics` — Prometheus
(латентности поиска и генерации, счётчики запросов/токенов/ошибок), сквозной
`X-Request-ID` в логах и заголовках ответа.

## Docker

`docker compose up -d --build` поднимает Qdrant, Ollama и приложение
(см. пометку в docker-compose.yml; модель в контейнер Ollama скачивается
отдельной командой).

Порты — по правилу урока 5.7: Ollama наружу не публикуется (к ней ходит только
приложение по внутренней сети compose; с хоста — `docker compose exec ollama ollama run …`),
Qdrant и приложение привязаны к `127.0.0.1`. Голая запись `"11434:11434"` открыла бы
LLM без аутентификации всей локальной сети — Docker публикует порты в обход файрвола.

## Структура

```
project/
├── app/
│   ├── main.py            # фабрика приложения, lifespan, DI через app.state
│   ├── config.py          # pydantic-settings, префикс APP_
│   ├── schemas.py         # строгие Pydantic-контракты запросов/ответов (ADR-0005)
│   ├── routers/           # /api/search, /api/generate(+history), /health,/ready
│   ├── services/          # chunking, embeddings, vectorstore, reranker, llm, rag, history
│   ├── middleware.py      # request-id, access-лог, метрики
│   └── metrics.py         # Prometheus
├── scripts/index_docs.py  # индексация data/docs в Qdrant (идемпотентная)
├── data/docs/             # 12 документов внутренней документации «Векторики»
├── tests/                 # unit + integration на фейках
└── docs/adr/              # архитектурные решения
```
