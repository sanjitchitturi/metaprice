"""Common helpers, metrics, and timing utilities."""

from __future__ import annotations

import time

import numpy as np


def rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute Root Mean Squared Error."""
    return float(np.sqrt(np.mean((predictions - targets) ** 2)))


def revenue_ratio(
    predicted_demand_curve: np.ndarray,
    true_demand_curve: np.ndarray,
    prices: np.ndarray,
) -> float:
    """Compute revenue ratio: model revenue / oracle revenue on query set.

    predicted_demand_curve and true_demand_curve are arrays aligned with prices.
    The model picks the price that maximizes predicted revenue, then we compute
    actual revenue at that price using the true demand.
    """
    predicted_revenue = prices * predicted_demand_curve
    best_idx = int(np.argmax(predicted_revenue))
    model_revenue = prices[best_idx] * true_demand_curve[best_idx]

    oracle_revenue_arr = prices * true_demand_curve
    oracle_revenue = float(np.max(oracle_revenue_arr))

    if oracle_revenue <= 0:
        return 0.0
    return float(model_revenue / oracle_revenue)


class Timer:
    """Simple context manager for timing blocks of code."""

    def __init__(self) -> None:
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0
