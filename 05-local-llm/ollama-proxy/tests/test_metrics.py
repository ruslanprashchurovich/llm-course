"""Окно метрик: перцентили и холодные старты — чистые юнит-тесты."""

from app.metrics import TpsWindow


def resp(tps: float, load_s: float = 0.01) -> dict:
    return {"eval_count": int(tps * 10), "eval_duration": 10_000_000_000,
            "load_duration": int(load_s * 1_000_000_000)}


def test_percentiles():
    w = TpsWindow()
    for tps in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        w.add_response(resp(tps))
    assert w.percentile(0.50) == 60
    assert w.percentile(0.95) == 100
    assert w.percentile(0.0) == 10


def test_empty_window_is_honest():
    assert TpsWindow().percentile(0.5) is None


def test_cold_starts_counted_in_recent_window():
    w = TpsWindow()
    for _ in range(10):
        w.add_response(resp(50, load_s=0.01))   # тёплые
    assert w.cold_starts() == 0
    w.add_response(resp(50, load_s=2.5))        # холодный
    w.add_response(resp(50, load_s=3.0))        # холодный
    assert w.cold_starts() == 2


def test_window_is_sliding():
    w = TpsWindow(maxlen=5)
    for tps in [10, 10, 10, 10, 10, 99, 99, 99, 99, 99]:
        w.add_response(resp(tps))
    assert w.percentile(0.0) == 99  # старые значения вытеснены
