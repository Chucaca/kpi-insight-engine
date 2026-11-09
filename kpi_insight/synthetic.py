"""Seeded synthetic data generator.

Simulates ~3 years of daily merchant KPIs for a multi-vertical POS SaaS
(verticals ``fnb``, ``retail``, ``spa``, ``hotel``) broken down by
province x sales channel. Everything here is fictional.

Accounting identity (exact, integer arithmetic, per series and in aggregate)::

    paying_t = paying_{t-1} + new_t - expiry_t + renewed_t
    active_merchant_t = round(paying_t * active_rate_true_t)   (rate re-derived as active / paying)

Injected scenarios (so the demo has something to find):

* ``SHOCK_PROVINCE``: active_rate ramps down by ~12 pp over the last month
  of data, in every vertical.
* ``SHOCK_CHANNEL``: new-merchant acquisition steps down 40% from the start of
  the same window, in every vertical.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

VERTICALS = ["fnb", "retail", "spa", "hotel"]
PROVINCES = [
    "Northbay", "Eastvale", "Harborview", "Lakeside", "Riverbend",
    "Highland", "Southport", "Westfield", "Pinecrest", "Sunridge",
]
CHANNELS = ["direct_sales", "partner", "telesales", "online_self_serve"]

START = "2023-11-01"
END = "2026-10-31"
SHOCK_PROVINCE = "Harborview"
SHOCK_CHANNEL = "partner"
SHOCK_START = "2026-10-01"

FLOW_KPIS = ["new_merchant", "expiry_merchant", "renewed_merchant"]
STOCK_KPIS = ["paying_merchant", "active_merchant"]
DIMENSIONS = ["province", "channel"]

# vertical-level parameters: base paying size, daily churn hazard, renewal share,
# baseline active rate, weekend lift on active rate, yearly growth of acquisition
_VERTICAL_PARAMS = {
    "fnb":    dict(size=60_000, hazard=0.0024, renew=0.40, rate=0.74, weekend=0.035, growth=0.10),
    "retail": dict(size=85_000, hazard=0.0020, renew=0.45, rate=0.70, weekend=0.02, growth=0.06),
    "spa":    dict(size=14_000, hazard=0.0028, renew=0.35, rate=0.66, weekend=0.04, growth=0.14),
    "hotel":  dict(size=6_000,  hazard=0.0018, renew=0.50, rate=0.62, weekend=0.03, growth=0.04),
}
# weekly profile for acquisition (Mon..Sun): fewer sign-ups on weekends
_NEW_DOW = np.array([1.15, 1.10, 1.05, 1.05, 1.00, 0.85, 0.80])
_EXPIRY_DOW = np.array([1.05, 1.00, 1.00, 1.00, 1.05, 0.95, 0.95])


@dataclass(frozen=True)
class SyntheticConfig:
    seed: int = 42
    start: str = START
    end: str = END
    shock_province: str = SHOCK_PROVINCE
    shock_channel: str = SHOCK_CHANNEL
    shock_start: str = SHOCK_START
    shock_rate_drop: float = 0.12
    shock_new_drop: float = 0.40


def _weights(rng: np.random.Generator, n: int, conc: float) -> np.ndarray:
    w = rng.dirichlet(np.full(n, conc))
    return np.sort(w)[::-1]


def generate(cfg: SyntheticConfig = SyntheticConfig()) -> pd.DataFrame:
    """Return a long DataFrame: one row per date x vertical x province x channel."""
    rng = np.random.default_rng(cfg.seed)
    dates = pd.date_range(cfg.start, cfg.end, freq="D")
    n_days = len(dates)
    dow = dates.dayofweek.to_numpy()
    t_years = np.arange(n_days) / 365.25
    shock_mask = dates >= pd.Timestamp(cfg.shock_start)
    # 0 -> 1 ramp across the shock window
    ramp = np.zeros(n_days)
    if shock_mask.any():
        k = shock_mask.sum()
        ramp[shock_mask] = np.linspace(1.0 / k, 1.0, k)

    frames = []
    for v in VERTICALS:
        p = _VERTICAL_PARAMS[v]
        w_prov = _weights(rng, len(PROVINCES), 2.0)
        w_chan = np.array([0.40, 0.30, 0.18, 0.12])
        # series index: province-major
        w = (w_prov[:, None] * w_chan[None, :]).ravel()
        prov_idx = np.repeat(np.arange(len(PROVINCES)), len(CHANNELS))
        chan_idx = np.tile(np.arange(len(CHANNELS)), len(PROVINCES))
        n_series = w.size

        paying0 = np.maximum(5, np.round(p["size"] * w)).astype(np.int64)
        hazard = p["hazard"] * rng.uniform(0.85, 1.15, n_series)
        # expected daily acquisition chosen so the base grows modestly
        new_base = paying0 * hazard * (1 - p["renew"]) * rng.uniform(1.05, 1.30, n_series)
        rate_base = np.clip(p["rate"] + rng.normal(0, 0.03, n_series), 0.4, 0.95)
        rate_trend = rng.normal(0.004, 0.006, n_series)  # per year

        shock_prov = PROVINCES.index(cfg.shock_province)
        shock_chan = CHANNELS.index(cfg.shock_channel)
        is_shock_prov = prov_idx == shock_prov
        is_shock_chan = chan_idx == shock_chan

        new = np.zeros((n_days, n_series), dtype=np.int64)
        exp = np.zeros_like(new)
        ren = np.zeros_like(new)
        pay = np.zeros_like(new)
        act = np.zeros_like(new)

        # slow-moving AR(1) noise on the active rate (shared + idiosyncratic)
        ar = np.zeros(n_series)
        prev = paying0.copy()
        exp_hist = np.zeros(n_series)  # EWMA of expiries, drives renewals
        for i in range(n_days):
            season = 1.0 + 0.08 * np.sin(2 * np.pi * (t_years[i] - 0.1))  # yearly cycle
            lam_new = new_base * (1 + p["growth"]) ** t_years[i] * _NEW_DOW[dow[i]] * season
            lam_new = np.where(is_shock_chan & shock_mask[i], lam_new * (1 - cfg.shock_new_drop), lam_new)
            new[i] = rng.poisson(lam_new)
            exp[i] = rng.binomial(prev, np.minimum(hazard * _EXPIRY_DOW[dow[i]], 0.5))
            exp_hist = 0.9 * exp_hist + 0.1 * exp[i] if i else exp[i].astype(float)
            ren[i] = rng.poisson(p["renew"] * exp_hist)
            pay[i] = prev + new[i] - exp[i] + ren[i]
            pay[i] = np.maximum(pay[i], 0)
            # keep the identity exact even if clipping happened (never in practice)
            exp[i] = prev + new[i] + ren[i] - pay[i]
            prev = pay[i]

            ar = 0.92 * ar + rng.normal(0, 0.004, n_series)
            weekend = p["weekend"] if dow[i] >= 5 else 0.0
            rate = rate_base + rate_trend * t_years[i] + weekend + ar
            rate = np.where(is_shock_prov, rate - cfg.shock_rate_drop * ramp[i], rate)
            rate = np.clip(rate, 0.05, 0.99)
            act[i] = np.rint(pay[i] * rate).astype(np.int64)

        idx = pd.MultiIndex.from_product([dates, range(n_series)], names=["date", "s"])
        df = pd.DataFrame(
            {
                "new_merchant": new.ravel(),
                "expiry_merchant": exp.ravel(),
                "renewed_merchant": ren.ravel(),
                "paying_merchant": pay.ravel(),
                "active_merchant": act.ravel(),
            },
            index=idx,
        ).reset_index()
        df.insert(1, "vertical", v)
        df.insert(2, "province", np.array(PROVINCES)[prov_idx][df["s"].to_numpy()])
        df.insert(3, "channel", np.array(CHANNELS)[chan_idx][df["s"].to_numpy()])
        # opening stock (day before START) so the identity can be checked on day 1
        df["_paying_open"] = 0
        first = df["date"] == dates[0]
        df.loc[first, "_paying_open"] = paying0[df.loc[first, "s"].to_numpy()]
        frames.append(df.drop(columns="s"))

    out = pd.concat(frames, ignore_index=True)
    for c in ["vertical", "province", "channel"]:
        out[c] = out[c].astype("category")
    return out


@lru_cache(maxsize=4)
def load(seed: int = 42) -> pd.DataFrame:
    """Cached default dataset (generation takes a couple of seconds)."""
    return generate(SyntheticConfig(seed=seed))


def add_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Derive ``active_rate`` = active / paying after any aggregation."""
    df = df.copy()
    df["active_rate"] = df["active_merchant"] / df["paying_merchant"].where(df["paying_merchant"] > 0)
    if "renewed_merchant" in df and "expiry_merchant" in df:
        df["renewal_ratio"] = df["renewed_merchant"] / df["expiry_merchant"].where(df["expiry_merchant"] > 0)
    return df


def daily_total(df: pd.DataFrame, vertical: str, by: list[str] | None = None) -> pd.DataFrame:
    """Aggregate one vertical to daily totals (optionally keeping dimensions)."""
    sub = df[df["vertical"] == vertical]
    keys = ["date"] + (by or [])
    cols = FLOW_KPIS + STOCK_KPIS
    g = sub.groupby(keys, observed=True)[cols].sum().reset_index()
    return add_rates(g)
