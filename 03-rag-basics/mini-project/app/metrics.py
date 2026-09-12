"""Метрики сервиса: скользящее окно латентностей + счётчики вердиктов (урок 3.7).

Мини-версия Prometheus-подхода: копим сырые числа, отчёт (p50/p95/p99)
считаем по требованию. Окно скользящее: метрика описывает последние N
запросов, а не всю жизнь процесса.
"""

from __future__ import annotations

from collections import Counter, deque


class MetricsWindow:
    def __init__(self, maxlen: int = 500):
        self.latencies: dict[str, deque] = {}
        self.verdicts: Counter = Counter()
        self.maxlen = maxlen

    def record(self, verdict: str, **stage_ms: float) -> None:
        self.verdicts[verdict] += 1
        for stage, value in stage_ms.items():
            self.latencies.setdefault(
                stage, deque(maxlen=self.maxlen)).append(float(value))

    @staticmethod
    def percentile(values, q: float) -> float:
        ordered = sorted(values)
        index = min(int(len(ordered) * q), len(ordered) - 1)
        return ordered[index]

    def summary(self) -> dict:
        report: dict = {"verdicts": dict(self.verdicts)}
        for stage, values in self.latencies.items():
            report[stage] = {
                "n": len(values),
                "mean": round(sum(values) / len(values), 1),
                "p50": round(self.percentile(values, 0.50), 1),
                "p95": round(self.percentile(values, 0.95), 1),
                "p99": round(self.percentile(values, 0.99), 1),
            }
        return report
