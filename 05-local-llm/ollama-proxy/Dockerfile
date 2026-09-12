# ollama-proxy — контейнер с прокси (урок 5.7).
# Сборка:  docker build -t ollama-proxy .
# Запуск удобнее через docker-compose.yml рядом (поднимает и Ollama).
FROM python:3.13-slim

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8005

# Один воркер: конкуренцию к модели всё равно ограничивает семафор,
# а окно метрик живёт в памяти процесса (несколько воркеров = несколько окон)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8005"]
