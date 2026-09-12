"""Сторожевой скрипт (watchdog) для Ollama (урок 5.5).

Раз в N секунд проверяет:
  * жив ли демон (GET /api/version)                       -> алерт OllamaDown
  * хватает ли свободной RAM                              -> алерт LowMemory
  * не залез ли хост в swap                               -> алерт SwapInUse
  * (опционально, реже) «глубокая» проба генерации:
    1 токен через /api/generate, замер скорости           -> алерт SlowGeneration

Алерты пишутся в stdout и в лог-файл. Правило «for-duration» встроено:
алерт срабатывает, только если условие держится for_seconds подряд, —
это защита от ложных срабатываний на единичных выбросах.

Запуск:
    python scripts/ollama_watchdog.py --interval 30 --min-free-gb 2
    python scripts/ollama_watchdog.py --probe-model qwen2.5:3b --probe-every 10
    python scripts/ollama_watchdog.py --max-iterations 2 --interval 1   # разовый прогон

Как сервис: Планировщик заданий Windows (при загрузке) / systemd-юнит / cron.
"""

from __future__ import annotations

import argparse
import logging
import time

import httpx
import psutil

NS = 1_000_000_000

logger = logging.getLogger("ollama_watchdog")


class Alert:
    """Алерт со счётчиком длительности: OK -> PENDING -> FIRING.

    Время передаётся параметром now — так класс тестируется без time.sleep
    (тот же приём, что тесты на фейках в уроке 3.8).
    """

    def __init__(self, name: str, for_seconds: float) -> None:
        self.name = name
        self.for_seconds = for_seconds
        self.violation_since: float | None = None
        self.firing = False

    def update(self, violated: bool, detail: str = "", now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        if not violated:
            if self.firing:
                logger.info("RESOLVED %s", self.name)
            self.violation_since = None
            self.firing = False
            return
        if self.violation_since is None:
            self.violation_since = now
        if not self.firing and now - self.violation_since >= self.for_seconds:
            self.firing = True
            logger.warning("FIRING %s: %s", self.name, detail)


def check_server(base_url: str, timeout: float = 5.0) -> bool:
    try:
        return httpx.get(f"{base_url}/api/version", timeout=timeout).status_code == 200
    except httpx.HTTPError:
        return False


def deep_probe(base_url: str, model: str, timeout: float = 180.0) -> dict | None:
    """Настоящая генерация на несколько токенов: проверяет весь путь, включая загрузку.

    ВАЖНО-1: проба сама грузит модель в RAM и сбрасывает таймер keep_alive —
    поэтому её надо запускать редко (--probe-every), а не на каждой итерации.
    ВАЖНО-2: num_predict >= 8 — на одном токене eval_duration близок к нулю
    и «скорость» получается бессмысленной (мы напоролись на tps=1000000).
    """
    payload = {
        "model": model,
        "prompt": "Перечисли несколько цветов радуги:",
        "stream": False,
        "options": {"num_predict": 8, "temperature": 0},
        "keep_alive": "5m",
    }
    try:
        resp = httpx.post(f"{base_url}/api/generate", json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        eval_count, eval_duration = data.get("eval_count", 0), data.get("eval_duration", 0)
        return {
            "load_seconds": data.get("load_duration", 0) / NS,
            "total_seconds": data.get("total_duration", 0) / NS,
            "tokens_per_second": eval_count / (eval_duration / NS) if eval_duration else None,
        }
    except httpx.HTTPError as exc:
        logger.error("deep probe failed: %s", exc)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Watchdog для Ollama")
    # 127.0.0.1, а не localhost: на Windows это минус ~1-2 c на соединение (урок 5.3)
    parser.add_argument("--url", default="http://127.0.0.1:11434")
    parser.add_argument("--interval", type=float, default=30.0, help="Период проверки, сек")
    parser.add_argument("--for-seconds", type=float, default=60.0,
                        help="Сколько секунд условие должно держаться до алерта")
    parser.add_argument("--min-free-gb", type=float, default=2.0,
                        help="Минимум свободной RAM, ГБ")
    parser.add_argument("--max-swap-percent", type=float, default=10.0,
                        help="Максимум занятого swap, %%")
    parser.add_argument("--probe-model", default=None,
                        help="Модель для глубокой пробы (по умолчанию — выключено)")
    parser.add_argument("--probe-every", type=int, default=10,
                        help="Глубокая проба раз в N итераций")
    parser.add_argument("--min-tps", type=float, default=3.0,
                        help="Минимальная скорость генерации, ток/с")
    parser.add_argument("--log-file", default="ollama_watchdog.log")
    parser.add_argument("--max-iterations", type=int, default=0,
                        help="Остановиться после N итераций (0 = работать вечно)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(args.log_file, encoding="utf-8")],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)  # не засорять лог каждым запросом

    alerts = {
        "down": Alert("OllamaDown", args.for_seconds),
        "memory": Alert("LowMemory", args.for_seconds),
        "swap": Alert("SwapInUse", args.for_seconds),
        "slow": Alert("SlowGeneration", 0),  # проба редкая, реагируем сразу
    }

    logger.info("watchdog запущен: url=%s interval=%ss", args.url, args.interval)
    iteration = 0
    while True:
        iteration += 1

        up = check_server(args.url)
        alerts["down"].update(not up, f"нет ответа от {args.url}/api/version")

        mem = psutil.virtual_memory()
        free_gb = mem.available / 1024**3
        alerts["memory"].update(free_gb < args.min_free_gb,
                                f"свободно {free_gb:.1f} ГБ < порога {args.min_free_gb} ГБ")

        swap = psutil.swap_memory()
        alerts["swap"].update(swap.percent > args.max_swap_percent,
                              f"swap занят на {swap.percent:.0f}%% — инференс мог уйти на диск")

        if up and args.probe_model and iteration % args.probe_every == 0:
            probe = deep_probe(args.url, args.probe_model)
            if probe is None:
                alerts["slow"].update(True, "глубокая проба не удалась")
            else:
                tps = probe["tokens_per_second"]
                logger.info("probe: load=%.1fs total=%.1fs tps=%s",
                            probe["load_seconds"], probe["total_seconds"],
                            f"{tps:.1f}" if tps else "n/a")
                alerts["slow"].update(bool(tps and tps < args.min_tps),
                                      f"скорость {tps:.1f} ток/с < порога {args.min_tps}" if tps else "")

        logger.info("итерация %d: up=%s free=%.1fGB swap=%.0f%%",
                    iteration, up, free_gb, swap.percent)
        if args.max_iterations and iteration >= args.max_iterations:
            logger.info("достигнут --max-iterations=%d, выходим", args.max_iterations)
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
