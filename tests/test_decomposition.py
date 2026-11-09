import numpy as np
import pandas as pd
import pytest

from kpi_insight import decomposition as dec
from kpi_insight.synthetic import FLOW_KPIS, daily_total

D1 = pd.Timestamp("2026-10-28")
D0 = dec.same_day_previous_month(D1)


def test_product_effects_sum_exactly():
    rng = np.random.default_rng(0)
    for _ in range(1000):
        b0, b1 = rng.integers(1, 10**6, 2).astype(float)
        c0, c1 = rng.uniform(0, 1, 2)
        e = dec.product_effects(b0, c0, b1, c1)
        assert e["b_effect"] + e["c_effect"] == pytest.approx(b1 * c1 - b0 * c0, rel=0, abs=1e-6)


def test_product_effects_integer_case_is_exact():
    e = dec.product_effects(100, 4, 120, 3)  # 400 -> 360
    assert e["b_effect"] == 70.0 and e["c_effect"] == -110.0
    assert e["b_effect"] + e["c_effect"] == e["delta"] == -40


def test_rate_mix_effects_sum_to_rate_change():
    w0, w1 = np.array([0.5, 0.3, 0.2]), np.array([0.4, 0.4, 0.2])
    r0, r1 = np.array([0.7, 0.6, 0.9]), np.array([0.65, 0.62, 0.95])
    out = dec.rate_mix_effects(w0, r0, w1, r1)
    assert out["contribution"].sum() == pytest.approx((w1 * r1).sum() - (w0 * r0).sum(), abs=1e-12)


def test_synthetic_stock_identity_is_exact(df):
    for (_, _, _), s in df.groupby(["vertical", "province", "channel"], observed=True):
        s = s.sort_values("date")
        prev = s["paying_merchant"].shift(1)
        prev.iloc[0] = s["_paying_open"].iloc[0]
        expected = prev + s["new_merchant"] - s["expiry_merchant"] + s["renewed_merchant"]
        assert (expected == s["paying_merchant"]).all()


def test_flow_decomposition_has_zero_residual(df, tree):
    daily = daily_total(df, "retail")
    out = dec.flow_decomposition(daily, D0, D1, tree)
    assert out["residual"] == 0
    assert sum(out["contributions"].values()) == out["delta"]
    assert set(out["contributions"]) == set(FLOW_KPIS)


def test_waterfall_steps_sum_to_change(df, tree):
    daily = daily_total(df, "fnb")
    wf = dec.driver_waterfall(daily, D0, D1, tree)
    steps = wf.loc[wf.kind == "step", "value"].sum()
    assert steps == pytest.approx(wf.value.iloc[-1] - wf.value.iloc[0], abs=1e-6)


@pytest.mark.parametrize("kpi", ["active_merchant", "active_rate", "new_merchant"])
@pytest.mark.parametrize("dim", ["province", "channel"])
def test_movers_sum_to_total_change(df, tree, kpi, dim):
    total = daily_total(df, "spa").set_index("date")
    dim_daily = daily_total(df, "spa", by=[dim])
    m = dec.mover_ranking(dim_daily, dim, kpi, D0, D1, tree)
    expected = dec.kpi_value(total, kpi, D1, tree) - dec.kpi_value(total, kpi, D0, tree)
    assert m["contribution"].sum() == pytest.approx(expected, abs=1e-9)
    if kpi == "active_merchant":
        eff = m["paying_merchant_effect"] + m["active_rate_effect"]
        assert np.allclose(eff, m["contribution"])


def test_movers_find_injected_shock(df, tree):
    dim_daily = daily_total(df, "fnb", by=["province"])
    m = dec.mover_ranking(dim_daily, "province", "active_rate", D0, D1, tree)
    assert dec.top_movers(m, 1)["member"].iloc[0] == "Harborview"


def test_scale_filter_excludes_small_members(tree):
    rows = []
    for d, vals in [(D0, (1000, 500, 5)), (D1, (1010, 490, 30))]:
        for member, v in zip("ABC", vals):
            rows.append(dict(date=d, province=member, active_merchant=v, paying_merchant=v * 2))
    m = dec.mover_ranking(pd.DataFrame(rows), "province", "active_merchant", D0, D1, tree)
    small = m.set_index("member").loc["C"]
    assert bool(small["filtered"])  # largest move (+25) but C is < 2% of the total
    assert "C" not in dec.top_movers(m)["member"].tolist()
    assert m["contribution"].sum() == 1530 - 1505  # filtering never changes the total


def test_same_day_previous_month_clips():
    assert dec.same_day_previous_month(pd.Timestamp("2026-03-31")) == pd.Timestamp("2026-02-28")
