"""Метрики прокси: окно скоростей (урок 5.5) + экспорт в Prometheus (урок 4.3)."""

from __future__ import annotations

from collections import deque

from prometheus_client import (CONTENT_TYPE_LATEST, CollectorRegistry, Counter,
                               Gauge, generate_latest)

NS = 1_000_000_000
COLD_START_S = 1.0  # загрузка дольше секунды считается холодным стартом


class TpsWindow:
    """Скользящее окно скоростей генерации — как в уроке 5.5."""

    def __init__(self, maxlen: int = 100):
        self.tps: deque[float] = deque(maxlen=maxlen)
        self.load_seconds: deque[float] = deque(maxlen=maxlen)

    def add_response(self, data: dict) -> None:
        if data.get("eval_duration"):
            self.tps.append(data["eval_count"] / (data["eval_duration"] / NS))
        self.load_seconds.append(data.get("load_duration", 0) / NS)

    def percentile(self, p: float) -> float | None:
        if not self.tps:
            return None
        ordered = sorted(self.tps)
        return ordered[min(int(len(ordered) * p), len(ordered) - 1)]

    def cold_starts(self, last_n: int = 10) -> int:
        recent = list(self.load_seconds)[-last_n:]
        return sum(1 for s in recent if s > COLD_START_S)


class ProxyMetrics:
    """Своя CollectorRegistry — не мусорим в глобальную (урок 5.5).

    Окно скоростей ведётся ПО МОДЕЛЯМ: p50 запасной (маленькой и быстрой) не должен
    маскировать деградацию основной — роутер судит только по окну основной.
    """

    def __init__(self, primary_model: str = "primary"):
        self.primary_model = primary_model
        self.windows: dict[str, TpsWindow] = {primary_model: TpsWindow()}
        self.registry = CollectorRegistry()
        self.requests_total = Counter(
            "proxy_requests_total", "Запросы к /generate по исходам",
            ["status"], registry=self.registry)
        self.routed_total = Counter(
            "proxy_routed_total", "Успешные ответы по маршруту: primary/fallback/canary",
            ["route"], registry=self.registry)
        self.g_up = Gauge("proxy_ollama_up", "Ollama отвечает на /api/version",
                          registry=self.registry)
        self.g_tps_p50 = Gauge("proxy_gen_tokens_per_second_p50",
                               "p50 скорости генерации основной модели по окну",
                               registry=self.registry)
        self.g_gpu_share = Gauge("proxy_model_gpu_share",
                                 "Доля основной модели в VRAM (0..1)", registry=self.registry)
        self.g_queue_ms = Gauge("proxy_last_queue_ms",
                                "Время последнего запроса в очереди, мс", registry=self.registry)
        self.g_degraded = Gauge("proxy_degraded",
                                "1 = новые запросы идут в запасную модель", registry=self.registry)

    @property
    def window(self) -> TpsWindow:
        """Окно основной модели — по нему healthz и роутер судят о здоровье."""
        return self.windows[self.primary_model]

    def window_for(self, model: str) -> TpsWindow:
        return self.windows.setdefault(model, TpsWindow())

    def reset_window(self, model: str) -> None:
        """Начать окно заново — после вердикта «медленно» ждём свежих (канареечных) ответов."""
        self.windows[model] = TpsWindow()

    def observe_response(self, data: dict, queue_ms: float, *, model: str | None = None,
                         route: str = "primary") -> None:
        self.window_for(model or self.primary_model).add_response(data)
        p50 = self.window.percentile(0.50)
        if p50 is not None:
            self.g_tps_p50.set(p50)
        self.g_queue_ms.set(queue_ms)
        self.requests_total.labels(status="ok").inc()
        self.routed_total.labels(route=route).inc()

    def observe_error(self, status: str) -> None:
        self.requests_total.labels(status=status).inc()

    def render(self) -> tuple[bytes, str]:
        return generate_latest(self.registry), CONTENT_TYPE_LATEST
