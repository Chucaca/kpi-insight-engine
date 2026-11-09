"""Robust trend and momentum.

* Trend: statsmodels RLM with Huber's T norm on ``y ~ 1 + t`` over a long
  window (default 180 days). Outliers such as holidays or data glitches get
  down-weighted instead of dragging the slope.
* Momentum: sklearn ``HuberRegressor`` over the last 90 days, compared with
  the long-window slope (accelerating / steady / decelerating / reversing).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass
class TrendResult:
    direction: str          # "up" | "down" | "flat"
    strength: str           # "strong" | "moderate" | "weak"
    slope_per_day: float    # units per day
    growth_pct_per_day: float  # slope / mean level, in percent
    window_change_pct: float   # slope * window / mean level, in percent
    t_value: float
    momentum: str
    momentum_slope_per_day: float
    window_days: int

    def to_dict(self) -> dict:
        return asdict(self)


def robust_slope(y: np.ndarray) -> tuple[float, float, float]:
    """Return ``(intercept, slope, t_value)`` of an RLM/HuberT fit on ``y ~ 1 + t``."""
    from statsmodels.robust.norms import HuberT
    from statsmodels.robust.robust_linear_model import RLM

    y = np.asarray(y, dtype=float)
    t = np.arange(len(y), dtype=float)
    X = np.column_stack([np.ones_like(t), t])
    if np.allclose(y, y[0]):
        return float(y[0]), 0.0, 0.0
    res = RLM(y, X, M=HuberT()).fit()
    slope = float(res.params[1])
    bse = float(res.bse[1]) if res.bse[1] > 0 else np.inf
    return float(res.params[0]), slope, slope / bse


def huber_slope(y: np.ndarray) -> float:
    """Slope (units/day) from sklearn's HuberRegressor; inputs are scaled for stability."""
    from sklearn.linear_model import HuberRegressor

    y = np.asarray(y, dtype=float)
    scale = np.std(y) or 1.0
    t = np.arange(len(y), dtype=float)
    tx = (t / max(len(y) - 1, 1)).reshape(-1, 1)
    model = HuberRegressor(epsilon=1.35, alpha=0.0, max_iter=500)
    model.fit(tx, (y - y.mean()) / scale)
    return float(model.coef_[0]) * scale / max(len(y) - 1, 1)


def analyze_trend(series: pd.Series, window: int = 180, momentum_window: int = 90) -> TrendResult:
    s = series.dropna()
    y = s.iloc[-window:].to_numpy(dtype=float)
    _, slope, tval = robust_slope(y)
    level = float(np.mean(y)) or 1.0
    growth = slope / level * 100
    window_change = slope * len(y) / level * 100

    if abs(tval) < 2.0:
        direction = "flat"
    else:
        direction = "up" if slope > 0 else "down"
    a = abs(window_change)
    strength = "strong" if a >= 5 else "moderate" if a >= 1.5 else "weak"

    m_slope = huber_slope(s.iloc[-momentum_window:].to_numpy(dtype=float))
    noise = abs(slope) + 1e-12
    if np.sign(m_slope) != np.sign(slope) and abs(m_slope) > 0.5 * noise and direction != "flat":
        momentum = "reversing"
    elif abs(m_slope) > 1.25 * abs(slope):
        momentum = "accelerating"
    elif abs(m_slope) < 0.75 * abs(slope):
        momentum = "decelerating"
    else:
        momentum = "steady"
    return TrendResult(
        direction=direction,
        strength=strength,
        slope_per_day=slope,
        growth_pct_per_day=growth,
        window_change_pct=window_change,
        t_value=tval,
        momentum=momentum,
        momentum_slope_per_day=m_slope,
        window_days=len(y),
    )


def fitted_line(series: pd.Series, window: int = 180) -> pd.Series:
    """Robust trend line over the last ``window`` points (for charts)."""
    s = series.dropna().iloc[-window:]
    b0, b1, _ = robust_slope(s.to_numpy(dtype=float))
    return pd.Series(b0 + b1 * np.arange(len(s)), index=s.index)
