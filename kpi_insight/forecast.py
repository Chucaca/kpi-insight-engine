"""Month-end forecasts by KPI type, plus a rolling-origin backtest.

KPI types (see ``configs/kpi_tree.json``):

* ``absolute``        stock KPI, value on the last day of the month.
                      Robust (RLM/HuberT) trend + day-of-week effects fitted on
                      the last ``window`` days, extrapolated to month-end and
                      re-anchored on the last week's residual.
* ``accumulate``      flow KPI, month sum. Month-to-date actuals + expected
                      remaining days, each remaining day = average of the same
                      weekday over the last ``lookback_weeks`` weeks.
* ``rate_accumulate`` ratio of two accumulated KPIs (forecast each, then divide).
* ``rate``            ratio of two stock KPIs (forecast each, then divide).

Formula forecast for the stock: ``paying_end = paying_today + fc_rem(new)
- fc_rem(expiry) + fc_rem(renewed)``, where ``fc_rem`` is the expected flow over
the remaining days of the month.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .driver_tree import DriverTree, load_tree


@dataclass
class Forecast:
    kpi: str
    type: str
    origin: str
    month_end: str
    value: float              # month-end forecast
    naive: float              # naive baseline forecast (last value / run-rate / MTD ratio)
    to_date: float            # current value (stock) or month-to-date (flow / ratio)
    days_elapsed: int
    days_remaining: int
    path: dict = field(default_factory=dict)  # date -> expected daily value (for charts)
    snaive: float = float("nan")  # seasonal naive: last observed value on the month-end's weekday

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("path")
        return d


def month_end(d: pd.Timestamp) -> pd.Timestamp:
    return (d + pd.offsets.MonthEnd(0)).normalize()


def _remaining_dates(d: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(d + pd.Timedelta(days=1), month_end(d), freq="D")


def forecast_absolute_path(s: pd.Series, d: pd.Timestamp, window: int = 90) -> pd.Series:
    """Expected daily values from d+1 to month-end for a stock series."""
    from statsmodels.robust.norms import HuberT
    from statsmodels.robust.robust_linear_model import RLM

    rem = _remaining_dates(d)
    if len(rem) == 0:
        return pd.Series(dtype=float)
    hist = s.loc[:d].dropna().iloc[-window:]
    n = len(hist)
    t = np.arange(n, dtype=float)

    def design(tt: np.ndarray, dates: pd.DatetimeIndex) -> np.ndarray:
        dow = np.zeros((len(tt), 6))
        dw = dates.dayofweek.to_numpy()
        for k in range(6):
            dow[:, k] = dw == k + 1
        return np.column_stack([np.ones(len(tt)), tt, dow])

    X = design(t, hist.index)
    y = hist.to_numpy(dtype=float)
    res = RLM(y, X, M=HuberT()).fit()
    resid_anchor = float(np.mean(y[-7:] - X[-7:] @ res.params))
    t_rem = n - 1 + np.arange(1, len(rem) + 1, dtype=float)
    pred = design(t_rem, rem) @ res.params + resid_anchor
    return pd.Series(pred, index=rem)


def forecast_accumulate_path(s: pd.Series, d: pd.Timestamp, lookback_weeks: int = 8) -> pd.Series:
    """Expected daily flow from d+1 to month-end via same-weekday averages."""
    rem = _remaining_dates(d)
    if len(rem) == 0:
        return pd.Series(dtype=float)
    hist = s.loc[d - pd.Timedelta(days=7 * lookback_weeks - 1): d]
    by_dow = hist.groupby(hist.index.dayofweek).mean()
    fallback = float(hist.mean())
    vals = [float(by_dow.get(x, fallback)) for x in rem.dayofweek]
    return pd.Series(vals, index=rem)


def _seasonal_naive(s: pd.Series, d: pd.Timestamp) -> float:
    """Most recent value (<= d) observed on the same weekday as the month-end."""
    target_dow = month_end(d).dayofweek
    hist = s.loc[d - pd.Timedelta(days=6): d]
    return float(hist[hist.index.dayofweek == target_dow].iloc[-1])


def _mtd(s: pd.Series, d: pd.Timestamp) -> float:
    return float(s.loc[d.replace(day=1): d].sum())


def forecast_kpi(daily: pd.DataFrame, kpi: str, d: pd.Timestamp, tree: DriverTree) -> Forecast:
    """Forecast one KPI's month-end value from data available up to ``d`` (inclusive)."""
    spec = tree.kpis[kpi]
    me = month_end(d)
    elapsed = d.day
    remaining = (me - d).days
    days_in_month = me.day

    if spec.type == "absolute":
        s = daily[kpi]
        path = forecast_absolute_path(s, d)
        cur = float(s.loc[d])
        value = float(path.iloc[-1]) if len(path) else cur
        return Forecast(kpi, spec.type, str(d.date()), str(me.date()), value, cur, cur,
                        elapsed, remaining, {str(k.date()): float(v) for k, v in path.items()},
                        snaive=_seasonal_naive(s, d))

    if spec.type == "accumulate":
        s = daily[kpi]
        path = forecast_accumulate_path(s, d)
        mtd = _mtd(s, d)
        value = mtd + float(path.sum())
        naive = mtd / elapsed * days_in_month
        return Forecast(kpi, spec.type, str(d.date()), str(me.date()), value, naive, mtd,
                        elapsed, remaining, {str(k.date()): float(v) for k, v in path.items()},
                        snaive=naive)

    num = forecast_kpi(daily, spec.numerator, d, tree)
    den = forecast_kpi(daily, spec.denominator, d, tree)
    value = num.value / den.value if den.value else float("nan")
    if spec.type == "rate_accumulate":
        naive = num.to_date / den.to_date if den.to_date else float("nan")
        to_date = snaive = naive
    else:  # "rate": ratio of two stock KPIs
        to_date = float(daily[kpi].loc[d])
        naive = to_date
        snaive = _seasonal_naive(daily[kpi], d)
    path = {k: num.path[k] / den.path[k] for k in num.path if den.path.get(k)}
    return Forecast(kpi, spec.type, str(d.date()), str(me.date()), value, naive, to_date,
                    elapsed, remaining, path, snaive=snaive)


def forecast_paying_formula(daily: pd.DataFrame, d: pd.Timestamp, tree: DriverTree,
                            target: str = "paying_merchant") -> Forecast:
    """Stock forecast from its flow relation: current + expected remaining net flows."""
    rel = tree.relation_for(target)
    cur = float(daily[target].loc[d])
    net = pd.Series(0.0, index=_remaining_dates(d))
    for k in rel.inflows:
        net = net.add(forecast_accumulate_path(daily[k], d), fill_value=0.0)
    for k in rel.outflows:
        net = net.sub(forecast_accumulate_path(daily[k], d), fill_value=0.0)
    path = cur + net.cumsum()
    value = float(path.iloc[-1]) if len(path) else cur
    me = month_end(d)
    return Forecast(f"{target}__formula", "formula", str(d.date()), str(me.date()), value, cur,
                    cur, d.day, (me - d).days, {str(k.date()): float(v) for k, v in path.items()},
                    snaive=_seasonal_naive(daily[target], d))


def forecast_all(daily: pd.DataFrame, d: pd.Timestamp, tree: DriverTree | None = None) -> dict[str, Forecast]:
    tree = tree or load_tree()
    daily = daily.set_index("date") if "date" in daily.columns else daily
    out = {k: forecast_kpi(daily, k, d, tree) for k in tree.kpis}
    for rel in tree.relations:
        if rel.kind == "flow":
            f = forecast_paying_formula(daily, d, tree, rel.target)
            out[f.kpi] = f
    return out


def actual_month_end(daily: pd.DataFrame, kpi: str, me: pd.Timestamp, tree: DriverTree) -> float:
    spec = tree.kpis[kpi]
    if spec.type in ("absolute", "rate"):
        return float(daily[kpi].loc[me])
    if spec.type == "accumulate":
        return float(daily[kpi].loc[me.replace(day=1): me].sum())
    num = actual_month_end(daily, spec.numerator, me, tree)
    den = actual_month_end(daily, spec.denominator, me, tree)
    return num / den


def backtest(df: pd.DataFrame, verticals: list[str] | None = None, n_months: int = 12,
             origin_days: tuple[int, ...] = (7, 14, 21), tree: DriverTree | None = None) -> pd.DataFrame:
    """Rolling-origin backtest over the last ``n_months`` complete month-ends.

    For each vertical, month and origin day, forecast the month-end value using
    only data up to the origin, then compare with the realised value.
    Returns one row per (vertical, kpi, month, origin) with APEs.
    """
    from .synthetic import VERTICALS, daily_total

    tree = tree or load_tree()
    verticals = verticals or VERTICALS
    rows = []
    for v in verticals:
        daily = daily_total(df, v).set_index("date")
        last = daily.index.max()
        ends = pd.date_range(end=last if last == month_end(last) else last - pd.offsets.MonthEnd(1),
                             periods=n_months, freq="ME")
        for me in ends:
            for od in origin_days:
                d = me.replace(day=od)
                fcs = forecast_all(daily, d, tree)
                for name, fc in fcs.items():
                    base = name.replace("__formula", "")
                    actual = actual_month_end(daily, base, me, tree)
                    rows.append(dict(vertical=v, kpi=name, type=fc.type, month_end=me.date(), origin_day=od,
                                     actual=actual, forecast=fc.value, naive=fc.naive,
                                     ape=abs(fc.value - actual) / abs(actual) * 100,
                                     snaive=fc.snaive,
                                     ape_naive=abs(fc.naive - actual) / abs(actual) * 100,
                                     ape_snaive=abs(fc.snaive - actual) / abs(actual) * 100))
    return pd.DataFrame(rows)


def backtest_summary(bt: pd.DataFrame) -> pd.DataFrame:
    g = bt.groupby(["kpi", "type"], sort=False).agg(
        mape=("ape", "mean"), mape_naive=("ape_naive", "mean"),
        mape_snaive=("ape_snaive", "mean"), n=("ape", "size"))
    g["improvement_vs_naive_pct"] = (1 - g["mape"] / g["mape_naive"]) * 100
    g["improvement_vs_best_baseline_pct"] = (1 - g["mape"] / g[["mape_naive", "mape_snaive"]].min(axis=1)) * 100
    return g.reset_index()
