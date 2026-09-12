# Модуль 3. RAG: Retrieval-Augmented Generation на практике

Учебный модуль курса «LLM для Python-разработчиков: от RAG до агентов».

Здесь вы соберёте полноценную вопросно-ответную систему по базе знаний вымышленной
IT-компании **«Векторика»** — той самой, по которой вы строили поиск в модуле 2
(копия документов лежит в `data/`): сначала «на коленке» из чистого
Python, затем — production-вариант на FastAPI + Qdrant + Ollama, с метриками качества,
защитой от галлюцинаций, агентами и тестами.

**Всё бесплатно и локально.** Никаких платных API: LLM крутится в Ollama на вашем CPU,
эмбеддинги считает sentence-transformers, векторная база Qdrant живёт в Docker.

## Чему вы научитесь

- Понимать механику RAG без магии фреймворков: чанкинг → эмбеддинги → поиск → генерация.
- Работать с Qdrant асинхронно и заворачивать RAG в FastAPI-эндпоинт.
- Измерять качество RAG: faithfulness, answer relevancy, context precision/recall (RAGAS + свои метрики).
- Бороться с галлюцинациями: пороги релевантности, цитирование источников, «я не знаю»-ответы.
- Писать агентов: function calling на чистом Python и stateful-агенты на LangGraph.
- Готовить LLM-сервис к продакшену: structlog, метрики латентности, санитизация ввода.

## Что понадобится

### Железо (честные цифры)

| Ресурс | Минимум | Комфортно |
|---|---|---|
| RAM | 8 ГБ | 16 ГБ |
| Диск | ~6 ГБ (модели + Docker-образы) | 10 ГБ |
| GPU | **не нужен** | — |

Скорость генерации LLM `qwen2.5:3b` на CPU (4–8 ядер): примерно **5–15 токенов/с**,
то есть развёрнутый ответ занимает 10–60 секунд. Для учёбы этого достаточно; в уроках
явно помечены «долгие» ячейки.

### Софт

1. **Python 3.13** (подойдёт и 3.12). Проверка: `python --version`.
2. **Docker Desktop** (для Qdrant, уроки 2+). На Windows нужен включённый WSL2.
3. **Ollama** — локальный сервер LLM: <https://ollama.com/download>. После установки
   он работает как фоновый сервис на `http://localhost:11434`.

## Установка

### Windows (PowerShell)

```powershell
cd C:\Users\<вы>\ml_projects\llm-course\03-rag-basics
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Если активация venv запрещена политикой:
`Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`.

### Linux / macOS / WSL (bash)

```bash
cd ~/ml_projects/llm-course/03-rag-basics
python3.13 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> **Если `torch` не ставится на Python 3.13** (нет собранного колеса под вашу платформу) —
> создайте venv на Python 3.12: `py -3.12 -m venv .venv`. Всё остальное не меняется.

### Модели

```powershell
# LLM (одна команда и для PowerShell, и для bash)
ollama pull qwen2.5:3b      # ~1.9 ГБ, основная модель модуля (умеет tool calling)
ollama pull qwen2.5:1.5b    # ~1 ГБ, запасной вариант для слабых ПК
```

Эмбеддинг-модель `intfloat/multilingual-e5-small` (~450 МБ) скачается с Hugging Face
автоматически при первом запуске урока 1 и закешируется в `~/.cache/huggingface`.

Если Hugging Face качается медленно, используйте зеркало:

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"   # PowerShell
```

```bash
export HF_ENDPOINT="https://hf-mirror.com"   # bash
```

### Qdrant (нужен с урока 2)

```powershell
docker compose up -d          # из папки модуля
```

Веб-интерфейс: <http://localhost:6333/dashboard>. Данные сохраняются в `./qdrant_storage`
(добавьте эту папку в `.gitignore`, если ведёте репозиторий).

### Запуск ноутбуков

```powershell
jupyter lab
```

Открывайте ноутбуки из корня модуля (важно для относительных путей к `data/`).

## Порядок уроков

| # | Файл | Тема | Результат |
|---|---|---|---|
| 1 | `01-rag-from-scratch.ipynb` | RAG с нуля: архитектура и первый пайплайн | Рабочий RAG на чистом Python + numpy |
| 2 | `02-qdrant-async.ipynb` | Qdrant и асинхронность | Async-пайплайн и FastAPI-эндпоинт `/ask` |
| 3 | `03-rag-evaluation.ipynb` | Оценка качества RAG | Golden-датасет, свои метрики, RAGAS локально |
| 4 | `04-hallucination-guardrails.ipynb` | Защита от галлюцинаций | Пороги, цитаты, self-check, «я не знаю» |
| 5 | `05-agents-tool-use.ipynb` | Агенты: LLM + инструменты | Цикл агента с function calling на чистом Python |
| 6 | `06-langgraph-stateful.ipynb` | Stateful-агенты на LangGraph | Граф с ToolNode, checkpointer, память диалога |
| 7 | `07-production-readiness.ipynb` | Подготовка к продакшену | structlog, латентность p50/p95, санитизация |
| 8 | `08-final-project.ipynb` | Финальный мини-проект | Сервис `mini-project/` + pytest |

Уроки последовательные: каждый опирается на предыдущие.

## Структура папки

```
03-rag-basics/
├── README.md
├── requirements.txt
├── docker-compose.yml        # Qdrant
├── data/                     # база знаний «Векторики» (14 txt, как в модуле 2)
├── utils.py                  # загрузка документов и чанкинг (из модуля 2)
├── 01-rag-from-scratch.ipynb ... 08-final-project.ipynb
├── service/                  # RAG-сервис урока 2: embedder, index_kb, FastAPI-app
├── eval/                     # golden.json — датасет для оценки качества (урок 3)
└── mini-project/             # финальный сервис (урок 8)
    ├── app/                  # FastAPI-приложение
    ├── tests/                # pytest (работают БЕЗ Docker и Ollama — на фейках)
    ├── pytest.ini
    └── README.md
```

## Мини-проект: быстрый старт

```powershell
docker compose up -d                      # Qdrant (из корня модуля)
cd mini-project
python -m app.ingest                      # индексация data/ в Qdrant
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Семантический бэкенд включается одной переменной окружения: `EMB_BACKEND=e5`
для обеих команд (отдельная коллекция создастся сама; подробности и все
настройки — в `mini-project/README.md`).

Тесты не требуют ни Docker, ни Ollama — зависимости подменяются фейками
(инъекция зависимостей, разбор — в уроке 8):

```powershell
cd mini-project
pytest -q
```

## Частые проблемы

- **Порт 11434 занят** — Ollama уже запущена как сервис, это нормально; отдельно стартовать её не нужно.
- **Docker не стартует** — включите WSL2: `wsl --install`, перезагрузка, затем Docker Desktop → Settings → General → «Use the WSL 2 based engine».
- **Ошибка длинных путей на Windows** при `pip install` — включите long paths: `git config --system core.longpaths true` и/или в реестре `HKLM\SYSTEM\CurrentControlSet\Control\FileSystem\LongPathsEnabled = 1`.
- **`OSError: [WinError 10061]` при обращении к Ollama** — сервис не запущен; откройте приложение Ollama или выполните `ollama serve`.
- **Медленная генерация** — переключитесь на `qwen2.5:1.5b`, закройте тяжёлые приложения; следите, чтобы модель не выгружалась (`ollama ps`).

## Полезные ссылки

- Ollama: <https://docs.ollama.com>
- Qdrant: <https://qdrant.tech/documentation/>
- sentence-transformers: <https://sbert.net>
- RAGAS: <https://docs.ragas.io>
- LangGraph: <https://langchain-ai.github.io/langgraph/>
- FastAPI: <https://fastapi.tiangolo.com>
