"""Настройка логирования приложения.

Один формат для всех логгеров, вывод в stdout — так логи собирает
и Docker, и systemd, и Kubernetes (12-factor: логи это поток событий).
"""

from __future__ import annotations

import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """Конфигурирует корневой логгер.

    ``force=True`` перезаписывает конфигурацию, даже если кто-то (uvicorn,
    pytest) уже успел вызвать basicConfig до нас.
    """
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        stream=sys.stdout,
        force=True,
    )
    # httpx на уровне INFO логирует каждый запрос — в проде это шум.
    logging.getLogger("httpx").setLevel(logging.WARNING)
