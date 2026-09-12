"""Маршрутизация между основной и запасной моделью по здоровью (капстоун 5.7).

Решение принимается там, где есть данные, — в healthz: он видит долю GPU из /api/ps
и p50 из окна метрик. /generate только читает готовое решение роутера, поэтому
каждый запрос не платит за лишний поход в /api/ps.

Правила (уроки 5.5, 5.6):
  * основная «съехала с GPU» — доля в VRAM ниже порога -> запросы в запасную;
  * p50 основной ниже degrade_p50_ratio от baseline -> запросы в запасную;
  * восстановление — по исчезновению признака. Для GPU это следующий healthz
    с нормальной долей (или модель выгружена — судить нечего). Для p50 сложнее:
    пока запросы идут в запасную, основная не отвечает и её окно не обновляется.
    Поэтому каждый canary_every-й запрос в режиме деградации идёт в основную
    как канарейка: несколько таких ответов — и healthz может пересудить.

Baseline берём из настроек; если не задан — выучиваем: первый p50 основной,
посчитанный по достаточному окну, пока признаков деградации нет.
Гистерезис и for-duration (урок 5.5) сюда не вошли намеренно — это упражнение.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RouteKind = Literal["primary", "fallback", "canary"]


@dataclass(frozen=True)
class RoutePick:
    model: str
    kind: RouteKind


@dataclass(frozen=True)
class RouteState:
    active_model: str
    degraded_to: str | None   # имя запасной, если новые запросы идут в неё
    reason: str | None
    baseline_tps: float | None


class ModelRouter:
    MIN_SAMPLES = 3  # столько ответов основной нужно, чтобы судить о p50 (и выучить baseline)

    def __init__(self, primary: str, fallback: str | None, *, baseline_tps: float | None,
                 gpu_share_floor: float, p50_ratio: float, canary_every: int = 5):
        self.primary = primary
        self.fallback = fallback or None
        self.baseline_tps = baseline_tps or None
        self.gpu_share_floor = gpu_share_floor
        self.p50_ratio = p50_ratio
        self.canary_every = canary_every
        self.gpu_degraded = False
        self.p50_degraded = False
        self._reasons: list[str] = []
        self._degraded_requests = 0

    # ------------------------------------------------------------- состояние
    @property
    def degraded(self) -> bool:
        return self.gpu_degraded or self.p50_degraded

    @property
    def degraded_to(self) -> str | None:
        return self.fallback if (self.degraded and self.fallback) else None

    @property
    def reason(self) -> str | None:
        return "; ".join(self._reasons) if self._reasons else None

    def state(self) -> RouteState:
        return RouteState(
            active_model=self.degraded_to or self.primary,
            degraded_to=self.degraded_to,
            reason=self.reason,
            baseline_tps=self.baseline_tps,
        )

    # ------------------------------------------------------------- решения
    def evaluate(self, *, gpu_share: float | None, p50_tps: float | None,
                 samples: int) -> RouteState:
        """Пересчитать вердикт по свежим наблюдениям за ОСНОВНОЙ моделью.

        gpu_share: доля основной в VRAM (None = не загружена, судить нечего);
        p50_tps / samples: p50 и размер окна скоростей основной модели.
        """
        reasons: list[str] = []

        # GPU: деградация по факту признака, восстановление — по его отсутствию
        self.gpu_degraded = gpu_share is not None and gpu_share < self.gpu_share_floor
        if self.gpu_degraded:
            reasons.append(f"основная модель лишь на {gpu_share:.0%} в VRAM "
                           f"(порог {self.gpu_share_floor:.0%}) — скорость упадёт (урок 5.6)")

        judgeable = p50_tps is not None and samples >= self.MIN_SAMPLES
        # baseline учим только на здоровой основной (иначе запомним больную)
        if self.baseline_tps is None and judgeable and not self.gpu_degraded:
            self.baseline_tps = p50_tps
        # p50 судим только при достаточном окне; иначе прежний вердикт остаётся (липкий)
        if self.baseline_tps and judgeable:
            self.p50_degraded = p50_tps < self.p50_ratio * self.baseline_tps
        if self.p50_degraded:
            shown = f"{p50_tps:.1f}" if p50_tps is not None else "n/a"
            reasons.append(f"p50 основной {shown} ток/с ниже {self.p50_ratio:.0%} "
                           f"от baseline {self.baseline_tps:.1f} (урок 5.5)")

        self._reasons = reasons
        if not self.degraded:
            self._degraded_requests = 0
        return self.state()

    def pick(self) -> RoutePick:
        """Куда отправить очередной запрос. Вызывается на каждый /generate."""
        if not self.degraded_to:
            return RoutePick(self.primary, "primary")
        self._degraded_requests += 1
        if self.canary_every and self._degraded_requests % self.canary_every == 0:
            return RoutePick(self.primary, "canary")  # даём основной шанс показать p50
        return RoutePick(self.fallback, "fallback")  # type: ignore[arg-type]
