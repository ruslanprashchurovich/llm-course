"""Лимит запросов на токен: скользящее окно в памяти (практика 5.7, средний уровень).

Почему в памяти и почему это ок: прокси — один процесс (max_concurrent=1, урок 5.3),
второй реплики нет, значит и общего хранилища счётчиков не нужно. Появится вторая
реплика — переезжайте на Redis, интерфейс check(key) останется тем же.

Почему скользящее окно, а не «счётчик, обнуляемый каждую минуту»: фиксированное окно
пропускает 2N запросов на стыке минут (N в 00:59 и ещё N в 01:00). Скользящее
гарантирует «не больше N за ЛЮБЫЕ 60 секунд» и умеет честно посчитать Retry-After:
слот освободится ровно тогда, когда самый старый запрос выпадет из окна.

Время приходит снаружи (clock) — тесты не спят (приём урока 5.5: Alert с инъекцией now).
"""

from __future__ import annotations

import math
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Decision:
    allowed: bool
    limit: int
    remaining: int       # сколько запросов ещё поместится в текущее окно
    retry_after_s: int   # > 0 только при отказе: через сколько секунд освободится слот


class RateLimiter:
    def __init__(self, limit_per_window: int, window_s: float = 60.0,
                 clock: Callable[[], float] = time.monotonic):
        self.limit = limit_per_window
        self.window_s = window_s
        self._clock = clock
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str) -> Decision:
        """Проверяет и, если можно, СРАЗУ учитывает запрос (check-and-consume).

        Разделять «проверить» и «учесть» нельзя: между ними пролезет второй запрос
        того же токена — классическая гонка на счётчике.
        """
        if self.limit <= 0:
            return Decision(allowed=True, limit=0, remaining=0, retry_after_s=0)  # выключен
        now = self._clock()
        hits = self._hits[key]
        # «Сброс окна» — это просто вытеснение старых отметок; отдельного таймера нет
        while hits and now - hits[0] >= self.window_s:
            hits.popleft()
        if len(hits) >= self.limit:
            retry_after = math.ceil(hits[0] + self.window_s - now)
            return Decision(allowed=False, limit=self.limit, remaining=0,
                            retry_after_s=max(retry_after, 1))
        hits.append(now)
        return Decision(allowed=True, limit=self.limit,
                        remaining=self.limit - len(hits), retry_after_s=0)
