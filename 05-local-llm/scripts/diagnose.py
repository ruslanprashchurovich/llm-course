"""Диагностика типовых проблем локальной LLM (урок 5.6).

Одноразовый прогон проверок с отчётом [OK]/[WARN]/[FAIL]:
  * Python и ОС
  * состояние RAM и swap хоста
  * доступность демона Ollama и его версия
  * список моделей и их размер против свободной RAM (влезет ли?)
  * что сейчас загружено в память (/api/ps) и какая доля на GPU
  * (опционально) проба генерации с замером холодного старта и скорости

Запуск:
    python scripts/diagnose.py
    python scripts/diagnose.py --probe qwen2.5:3b
"""

from __future__ import annotations

import argparse
import platform
import sys
import time

import httpx
import psutil

NS = 1_000_000_000
GB = 1024**3


def report(status: str, message: str) -> None:
    print(f"[{status:^4}] {message}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Диагностика Ollama и хоста")
    # 127.0.0.1, а не localhost: на Windows это минус ~1-2 c на соединение (урок 5.3)
    parser.add_argument("--url", default="http://127.0.0.1:11434")
    parser.add_argument("--probe", default=None, metavar="MODEL",
                        help="Выполнить пробную генерацию указанной моделью")
    args = parser.parse_args()

    print("=== Диагностика локальной LLM ===\n")

    # --- 1. Окружение -------------------------------------------------------
    report("OK", f"Python {platform.python_version()} на {platform.system()} {platform.release()}")

    # --- 2. Хост: RAM и swap -------------------------------------------------
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    report("OK" if mem.available > 2 * GB else "WARN",
           f"RAM: всего {mem.total / GB:.1f} ГБ, свободно {mem.available / GB:.1f} ГБ")
    if swap.percent > 25:
        report("WARN", f"Swap занят на {swap.percent:.0f}% — если модель уехала в swap, "
                       "скорость падает в 10-100 раз. Перезапустите Ollama или уменьшите модель.")
    else:
        report("OK", f"Swap: занят на {swap.percent:.0f}%")

    # --- 3. Демон Ollama -----------------------------------------------------
    try:
        version = httpx.get(f"{args.url}/api/version", timeout=3).json().get("version", "?")
        report("OK", f"Ollama отвечает на {args.url}, версия {version}")
    except httpx.HTTPError as exc:
        report("FAIL", f"Ollama не отвечает на {args.url}: {exc.__class__.__name__}")
        print("\nЧто проверить: запущен ли демон (ollama serve / служба / контейнер), "
              "тот ли порт, не блокирует ли файрвол.")
        return 1

    # --- 4. Модели на диске против свободной RAM ----------------------------
    try:
        models = httpx.get(f"{args.url}/api/tags", timeout=5).json().get("models", [])
    except httpx.HTTPError:
        models = []
    if not models:
        report("WARN", "Нет скачанных моделей (ollama pull <model>)")
    for m in models:
        size_gb = m.get("size", 0) / GB
        name = m.get("name", "?")
        # Эмпирика урока 5.2: рабочий объём ~= файл * 1.1 + ~1 ГБ (KV-кеш + runtime)
        needed = size_gb * 1.1 + 1.0
        if needed <= mem.available / GB:
            report("OK", f"{name}: файл {size_gb:.1f} ГБ, нужно ~{needed:.1f} ГБ — помещается")
        else:
            report("WARN", f"{name}: файл {size_gb:.1f} ГБ, нужно ~{needed:.1f} ГБ, "
                           f"а свободно {mem.available / GB:.1f} ГБ — риск OOM/swap")

    # --- 5. Что загружено сейчас и где (GPU/CPU) ------------------------------
    try:
        loaded = httpx.get(f"{args.url}/api/ps", timeout=5).json().get("models", [])
        if loaded:
            for m in loaded:
                share = m.get("size_vram", 0) / m.get("size", 1)
                where = "GPU" if share > 0.99 else ("CPU" if share < 0.01 else f"GPU {share:.0%}")
                status = "OK" if share > 0.99 or share < 0.01 else "WARN"
                report(status, f"В памяти: {m.get('name')} ({m.get('size', 0) / GB:.1f} ГБ, {where}), "
                               f"выгрузится: {m.get('expires_at', '?')}")
                if 0.01 <= share <= 0.99:
                    report("WARN", "Модель загружена ЧАСТИЧНО на GPU — скорость будет между "
                                   "GPU и CPU. Кто-то занял VRAM? (урок 5.6, болезнь 2)")
        else:
            report("OK", "Сейчас ни одна модель не загружена в память (холодный старт впереди)")
    except httpx.HTTPError:
        report("WARN", "/api/ps недоступен (старая версия Ollama?)")

    # --- 6. Процессы ollama ---------------------------------------------------
    procs = [p for p in psutil.process_iter(["name", "memory_info"])
             if p.info["name"] and "ollama" in p.info["name"].lower()]
    for p in procs:
        report("OK", f"Процесс {p.info['name']} (pid {p.pid}): "
                     f"RSS {p.info['memory_info'].rss / GB:.2f} ГБ")

    # --- 7. Проба генерации ---------------------------------------------------
    if args.probe:
        print(f"\nПробная генерация моделью {args.probe} (до 3 минут)...")
        payload = {
            "model": args.probe,
            "prompt": "Ответь одним словом: какого цвета небо днём?",
            "stream": False,
            "options": {"num_predict": 16},
        }
        t0 = time.perf_counter()
        try:
            resp = httpx.post(f"{args.url}/api/generate", json=payload, timeout=180)
            resp.raise_for_status()
            data = resp.json()
            load_s = data.get("load_duration", 0) / NS
            eval_count = data.get("eval_count", 0)
            eval_duration = data.get("eval_duration", 0)
            tps = eval_count / (eval_duration / NS) if eval_duration else 0
            report("OK", f"Ответ за {time.perf_counter() - t0:.1f} с "
                         f"(загрузка модели {load_s:.1f} с, генерация {tps:.1f} ток/с)")
            if load_s > 30:
                report("WARN", "Загрузка модели > 30 с — модель на медленном диске (HDD?) "
                               "или память под давлением")
            if 0 < tps < 3:
                report("WARN", "Меньше 3 ток/с — проверьте swap, троттлинг CPU, фоновую нагрузку "
                               "(чек-лист — в уроке 5.6)")
        except httpx.HTTPError as exc:
            report("FAIL", f"Проба не удалась: {exc}")
            return 1

    print("\nДиагностика завершена.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
