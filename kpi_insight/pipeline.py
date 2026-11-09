"""Glue: data -> trend / forecast / decomposition -> a JSON-serialisable facts dict.

The facts dict is the single source of truth for the narrative layer: the LLM
(or the template) may only quote numbers that appear in it.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import decomposition as dec
from .driver_tree import DriverTree, load_tree
from .forecast import Forecast, forecast_all
from .synthetic import VERTICALS, daily_total, load
from .trend import analyze_trend


@dataclass
class Analysis:
    vertical: str
    date: pd.Timestamp
    compare_date: pd.Timestamp
    daily: pd.DataFrame              # indexed by date, vertical total
    lfl: pd.DataFrame
    trends: dict
    forecasts: dict[str, Forecast]
    waterfall: pd.DataFrame
    flows: dict
    movers: dict[str, pd.DataFrame]
    facts: dict


def _r(x: float, nd: int = 1) -> float:
    return round(float(x), nd)


def _fmt_value(kpi: str, x: float, tree: DriverTree) -> float:
    """Merchant counts -> int, ratios -> percent with 1 decimal."""
    if tree.kpis[kpi].unit == "ratio":
        return _r(x * 100, 1)
    return int(round(x))


def analyze(vertical: str, date: str | pd.Timestamp, df: pd.DataFrame | None = None,
            tree: DriverTree | None = None, top_n: int = 5) -> Analysis:
    tree = tree or load_tree()
    df = load() if df is None else df
    if vertical not in VERTICALS:
        raise ValueError(f"unknown vertical {vertical!r}; expected one of {VERTICALS}")
    d = pd.Timestamp(date).normalize()
    daily = daily_total(df, vertical).set_index("date")
    if d not in daily.index or d - pd.Timedelta(days=180) < daily.index.min():
        raise KeyError(f"date {d.date()} outside data range "
                       f"[{(daily.index.min() + pd.Timedelta(days=180)).date()}, {daily.index.max().date()}]")
    d0 = dec.same_day_previous_month(d)
    hist = daily.loc[:d]

    lfl = dec.like_for_like(hist, d, tree)
    trend_kpis = [k for k, s in tree.kpis.items() if s.type in ("absolute", "rate")]
    trends = {k: analyze_trend(hist[k]) for k in trend_kpis}
    fcs = forecast_all(hist, d, tree)
    wf = dec.driver_waterfall(hist, d0, d, tree)
    flow_target = next(r.target for r in tree.relations if r.kind == "flow")
    flows = dec.flow_decomposition(hist, d0, d, tree, flow_target)

    headline = tree.headline_kpi
    rate_kpi = tree.relation_for(headline).factors[1]
    flow_rel = tree.relation_for(flow_target)
    # headline, its rate factor, and the first inflow (acquisition)
    mover_kpis = [headline, rate_kpi] + list(flow_rel.inflows[:1])
    movers = {}
    for dim in tree.dimensions:
        dim_daily = daily_total(df, vertical, by=[dim])
        dim_daily = dim_daily[dim_daily["date"] <= d]
        for kpi in mover_kpis:
            movers[f"{kpi}|{dim}"] = dec.mover_ranking(dim_daily, dim, kpi, d0, d, tree)

    facts = build_facts(vertical, d, d0, tree, lfl, trends, fcs, wf, flows, movers, top_n)
    return Analysis(vertical, d, d0, hist, lfl, trends, fcs, wf, flows, movers, facts)


def build_facts(vertical, d, d0, tree, lfl, trends, fcs, wf, flows, movers, top_n) -> dict:
    headline = tree.headline_kpi
    kpis = {}
    for row in lfl.itertuples():
        f = fcs[row.kpi]
        kpis[row.kpi] = {
            "label": row.label,
            "type": row.type,
            "unit": "percent" if tree.kpis[row.kpi].unit == "ratio" else "merchants",
            "basis": "level" if row.type in ("absolute", "rate") else "month-to-date",
            "current": _fmt_value(row.kpi, row.current, tree),
            "previous_month_same_day": _fmt_value(row.kpi, row.previous, tree),
            "change": (_r((row.current - row.previous) * 100, 1) if tree.kpis[row.kpi].unit == "ratio"
                       else int(round(row.delta))),
            "change_unit": "pp" if tree.kpis[row.kpi].unit == "ratio" else "merchants",
            "change_pct": _r(row.delta_pct, 1),
            "month_end_forecast": _fmt_value(row.kpi, f.value, tree),
            "good_direction": tree.kpis[row.kpi].good_direction,
        }
        if row.kpi in trends:
            t = trends[row.kpi]
            kpis[row.kpi]["trend"] = {
                "direction": t.direction, "strength": t.strength, "momentum": t.momentum,
                "window_days": t.window_days,
                "growth_pct_per_day": _r(t.growth_pct_per_day, 3),
                "change_over_window_pct": _r(t.window_change_pct, 1),
            }
    formula = next((v for k, v in fcs.items() if k.endswith("__formula")), None)
    if formula is not None:
        base = formula.kpi.replace("__formula", "")
        kpis[base]["month_end_forecast_formula"] = int(round(formula.value))

    waterfall = [{"label": r.label, "kind": r.kind, "value": int(round(r.value))} for r in wf.itertuples()]
    # make the rounded steps add up exactly to the rounded end - start (largest step absorbs it)
    steps = [w for w in waterfall if w["kind"] == "step"]
    gap = (waterfall[-1]["value"] - waterfall[0]["value"]) - sum(w["value"] for w in steps)
    if steps and gap:
        max(steps, key=lambda w: abs(w["value"]))["value"] += gap
    flow = {
        "target": flows["target"], "start": int(flows["start"]), "end": int(flows["end"]),
        "change": int(flows["delta"]),
        "contributions": {k: int(v) for k, v in flows["contributions"].items()},
    }

    mv = {}
    for key, m in movers.items():
        kpi, dim = key.split("|")
        is_rate = tree.kpis[kpi].unit == "ratio"
        top = dec.top_movers(m, top_n)
        rows = []
        for r in top.itertuples():
            item = {
                "member": str(r.member),
                "contribution": _r(r.contribution * 100, 2) if is_rate else int(round(r.contribution)),
                "share_of_change_pct": _r(r.share_of_delta * 100, 1),
                "share_of_gross_movement_pct": _r(r.share_of_gross * 100, 1),
                "value_prev": _r(r.value_prev * 100, 1) if is_rate else int(round(r.value_prev)),
                "value_curr": _r(r.value_curr * 100, 1) if is_rate else int(round(r.value_curr)),
                "size_share_pct": _r(r.scale_share * 100, 1),
            }
            if is_rate:
                item["own_rate_effect"] = _r(r.rate_effect * 100, 2)
                item["mix_effect"] = _r(r.mix_effect * 100, 2)
            else:
                for col in m.columns:
                    if col.endswith("_effect"):
                        item[col] = int(round(getattr(r, col)))
                if hasattr(r, "rate_prev"):
                    item["rate_prev_pct"] = _r(r.rate_prev * 100, 1)
                    item["rate_curr_pct"] = _r(r.rate_curr * 100, 1)
            rows.append(item)
        mv[key] = {
            "kpi": kpi, "dimension": dim,
            "unit": "pp" if is_rate else "merchants",
            "total_change": _r(m.attrs["total_delta"] * 100, 2) if is_rate else int(round(m.attrs["total_delta"])),
            "filtered_members": int(m["filtered"].sum()),
            "gross_movement": (_r(m["contribution"].abs().sum() * 100, 2) if is_rate
                               else int(round(m["contribution"].abs().sum()))),
            "top": rows,
        }

    return {
        "vertical": vertical,
        "date": str(d.date()),
        "compare_date": str(d0.date()),
        "headline_kpi": headline,
        "month_end": fcs[headline].month_end,
        "month_end_weekday": pd.Timestamp(fcs[headline].month_end).day_name(),
        "days_remaining_in_month": fcs[headline].days_remaining,
        "kpis": kpis,
        "headline_waterfall": waterfall,
        "flow_decomposition": flow,
        "movers": mv,
        "notes": [
            "Comparisons are like-for-like: same day of the previous month.",
            "Flow KPIs (new, expiry, renewed) are month-to-date sums.",
            f"Movers below {tree.mover_min_scale_share * 100:g}% of total size are ignored.",
        ],
    }


def available_dates(vertical: str = "fnb", df: pd.DataFrame | None = None) -> tuple[str, str]:
    df = load() if df is None else df
    dates = df.loc[df["vertical"] == vertical, "date"]
    return str((dates.min() + pd.Timedelta(days=180)).date()), str(dates.max().date())
