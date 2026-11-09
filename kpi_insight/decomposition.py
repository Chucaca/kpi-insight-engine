"""Root-cause decomposition.

(a) ``product_effects``   midpoint (two-factor Shapley) split of the change in
    ``A = B x C`` into a B-effect and a C-effect; they sum to dA exactly
    (algebraically; numerically up to float rounding).
(b) ``flow_decomposition`` additive split of a stock change into its flows.
(c) ``like_for_like``     today vs the same day of the previous month
    (month-to-date for flow KPIs).
(d) ``mover_ranking``     each dimension member's contribution to the total
    change, its share of the total, with a scale filter that ignores members
    smaller than ``min_scale_share`` of the total.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .driver_tree import DriverTree


# ---------------------------------------------------------------- (a) product
def product_effects(b0: float, c0: float, b1: float, c1: float) -> dict[str, float]:
    """Midpoint decomposition of d(B*C).

    B-effect = dB * (C0 + C1) / 2,  C-effect = dC * (B0 + B1) / 2.
    This equals the Shapley value of the two-player game and has no residual:
    B-effect + C-effect = B1*C1 - B0*C0.
    """
    b_eff = (b1 - b0) * (c0 + c1) / 2.0
    c_eff = (c1 - c0) * (b0 + b1) / 2.0
    return {"a0": b0 * c0, "a1": b1 * c1, "delta": b1 * c1 - b0 * c0,
            "b_effect": b_eff, "c_effect": c_eff}


def rate_mix_effects(w0: np.ndarray, r0: np.ndarray, w1: np.ndarray, r1: np.ndarray) -> pd.DataFrame:
    """Split the change of an aggregate rate R = sum(w_i * r_i) by member.

    ``w`` are weight shares (sum to 1), ``r`` member rates. Each member gets a
    mix effect (its weight moved) and a rate effect (its own rate moved),
    using the same midpoint rule, so the members sum exactly to dR.
    """
    mix = (w1 - w0) * (r0 + r1) / 2.0
    rate = (r1 - r0) * (w0 + w1) / 2.0
    return pd.DataFrame({"mix_effect": mix, "rate_effect": rate, "contribution": mix + rate})


# ---------------------------------------------------------------- (b) flows
def flow_decomposition(daily: pd.DataFrame, d0: pd.Timestamp, d1: pd.Timestamp,
                       tree: DriverTree, target: str = "paying_merchant") -> dict:
    """Stock change between d0 and d1 = sum of flows over (d0, d1]."""
    daily = daily.set_index("date") if "date" in daily.columns else daily
    rel = tree.relation_for(target)
    window = daily.loc[d0 + pd.Timedelta(days=1): d1]
    parts = {k: float(window[k].sum()) for k in rel.inflows}
    parts.update({k: -float(window[k].sum()) for k in rel.outflows})
    start, end = float(daily.loc[d0, target]), float(daily.loc[d1, target])
    total = math.fsum(parts.values())
    return {"target": target, "start": start, "end": end, "delta": end - start,
            "contributions": parts, "residual": (end - start) - total}


# ---------------------------------------------------------------- (c) like for like
def same_day_previous_month(d: pd.Timestamp) -> pd.Timestamp:
    """Same day-of-month one month earlier (clipped, e.g. 31 Mar -> 28/29 Feb)."""
    return d - pd.DateOffset(months=1)


def kpi_value(daily: pd.DataFrame, kpi: str, d: pd.Timestamp, tree: DriverTree) -> float:
    """Level for stock KPIs, month-to-date for flow KPIs, ratio of MTDs for rate_accumulate."""
    spec = tree.kpis[kpi]
    if spec.type in ("absolute", "rate"):
        return float(daily.loc[d, kpi])
    if spec.type == "accumulate":
        return float(daily.loc[d.replace(day=1): d, kpi].sum())
    num = kpi_value(daily, spec.numerator, d, tree)
    den = kpi_value(daily, spec.denominator, d, tree)
    return num / den if den else float("nan")


def like_for_like(daily: pd.DataFrame, d: pd.Timestamp, tree: DriverTree) -> pd.DataFrame:
    daily = daily.set_index("date") if "date" in daily.columns else daily
    d0 = same_day_previous_month(d)
    rows = []
    for k, spec in tree.kpis.items():
        cur, prev = kpi_value(daily, k, d, tree), kpi_value(daily, k, d0, tree)
        rows.append({"kpi": k, "label": spec.label, "type": spec.type, "date": str(d.date()),
                     "compare_date": str(d0.date()), "current": cur, "previous": prev,
                     "delta": cur - prev, "delta_pct": (cur / prev - 1) * 100 if prev else float("nan")})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- headline waterfall
def driver_waterfall(daily: pd.DataFrame, d0: pd.Timestamp, d1: pd.Timestamp, tree: DriverTree) -> pd.DataFrame:
    """Waterfall of the headline product KPI from d0 to d1.

    ``A = B x C`` is split with the midpoint rule; the B-effect is further split
    in proportion to B's flows (exact, because the B-effect is linear in dB).
    Rows: start, one per flow, C-effect, end. Steps sum exactly to end - start.
    """
    daily = daily.set_index("date") if "date" in daily.columns else daily
    rel = tree.relation_for(tree.headline_kpi)
    b, c = rel.factors
    b0, b1 = float(daily.loc[d0, b]), float(daily.loc[d1, b])
    c0, c1 = float(daily.loc[d0, c]), float(daily.loc[d1, c])
    eff = product_effects(b0, c0, b1, c1)
    steps = [("start", tree.kpis[tree.headline_kpi].label + f" {d0.date()}", eff["a0"])]
    flow_rel = tree.relation_for(b)
    if flow_rel is not None and flow_rel.kind == "flow":
        fd = flow_decomposition(daily, d0, d1, tree, b)
        avg_c = (c0 + c1) / 2.0
        for k, v in fd["contributions"].items():
            steps.append(("step", f"{tree.kpis[k].label} (via {tree.kpis[b].label.lower()})", v * avg_c))
    else:
        steps.append(("step", f"{tree.kpis[b].label} effect", eff["b_effect"]))
    steps.append(("step", f"{tree.kpis[c].label} effect", eff["c_effect"]))
    steps.append(("end", tree.kpis[tree.headline_kpi].label + f" {d1.date()}", eff["a1"]))
    return pd.DataFrame(steps, columns=["kind", "label", "value"])


# ---------------------------------------------------------------- (d) movers
def _member_values(dim_daily: pd.DataFrame, dim: str, kpi: str, d: pd.Timestamp, tree: DriverTree) -> pd.Series:
    spec = tree.kpis[kpi]
    if spec.type in ("absolute", "rate"):
        snap = dim_daily[dim_daily["date"] == d]
    else:
        snap = dim_daily[(dim_daily["date"] >= d.replace(day=1)) & (dim_daily["date"] <= d)]
    return snap.groupby(dim, observed=True)[[c for c in snap.columns if c not in ("date", dim)]].sum()


def mover_ranking(dim_daily: pd.DataFrame, dim: str, kpi: str, d0: pd.Timestamp, d1: pd.Timestamp,
                  tree: DriverTree, min_scale_share: float | None = None) -> pd.DataFrame:
    """Rank members of ``dim`` by their contribution to the change of ``kpi`` (d0 -> d1).

    * Additive KPIs (absolute / accumulate): contribution = member delta.
      For a product KPI the contribution is also split into factor effects.
    * Rate KPIs: contribution = member rate effect + mix effect (sums to dR).

    ``share_of_delta`` = contribution / total (net) delta; it explodes when
    positives and negatives cancel, so ``share_of_gross`` = |contribution| /
    sum(|contributions|) is reported as well. ``scale_share`` = the
    member's size relative to the total; members below ``min_scale_share``
    (default from config, 2%) are flagged ``filtered`` and excluded from the
    ranking because tiny bases produce noisy, misleading swings.
    """
    min_scale_share = tree.mover_min_scale_share if min_scale_share is None else min_scale_share
    spec = tree.kpis[kpi]
    v0 = _member_values(dim_daily, dim, kpi, d0, tree)
    v1 = _member_values(dim_daily, dim, kpi, d1, tree)
    members = v0.index.union(v1.index)
    v0, v1 = v0.reindex(members, fill_value=0), v1.reindex(members, fill_value=0)

    if spec.type in ("rate", "rate_accumulate"):
        num, den = spec.numerator, spec.denominator
        w0 = (v0[den] / v0[den].sum()).to_numpy()
        w1 = (v1[den] / v1[den].sum()).to_numpy()
        r0 = (v0[num] / v0[den].where(v0[den] > 0)).fillna(0).to_numpy()
        r1 = (v1[num] / v1[den].where(v1[den] > 0)).fillna(0).to_numpy()
        out = rate_mix_effects(w0, r0, w1, r1)
        out.index = members
        out.insert(0, "value_prev", r0)
        out.insert(1, "value_curr", r1)
        out["scale_share"] = np.maximum(w0, w1)
    else:
        x0, x1 = v0[kpi].astype(float), v1[kpi].astype(float)
        out = pd.DataFrame({"value_prev": x0, "value_curr": x1, "contribution": x1 - x0})
        rel = tree.relation_for(kpi)
        if rel is not None and rel.kind == "product":
            b, c = rel.factors
            b0, b1 = v0[b].astype(float), v1[b].astype(float)
            # member-level factor C (e.g. active_rate) = A / B
            c0 = (x0 / b0.where(b0 > 0)).fillna(0)
            c1 = (x1 / b1.where(b1 > 0)).fillna(0)
            eff = product_effects(b0, c0, b1, c1)
            out[f"{b}_effect"] = eff["b_effect"]
            out[f"{c}_effect"] = eff["c_effect"]
            out["rate_prev"], out["rate_curr"] = c0, c1
        scale = np.maximum(x0.abs(), x1.abs())
        out["scale_share"] = scale / max(x0.abs().sum(), x1.abs().sum(), 1e-12)

    total = float(out["contribution"].sum())
    out["share_of_delta"] = out["contribution"] / total if total else np.nan
    gross = float(out["contribution"].abs().sum())
    out["share_of_gross"] = out["contribution"].abs() / gross if gross else np.nan
    out["filtered"] = out["scale_share"] < min_scale_share
    out.index.name = "member"
    out = out.reset_index()
    out.insert(0, "dimension", dim)
    out.insert(1, "kpi", kpi)
    out = out.reindex(out["contribution"].abs().sort_values(ascending=False).index)
    out.attrs["total_delta"] = total
    return out.reset_index(drop=True)


def top_movers(movers: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    return movers[~movers["filtered"]].head(n)
