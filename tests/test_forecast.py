import pandas as pd
import pytest

from kpi_insight.forecast import backtest, backtest_summary, forecast_all, month_end
from kpi_insight.synthetic import daily_total


@pytest.fixture(scope="module")
def daily(df):
    return daily_total(df, "fnb").set_index("date")


def test_forecast_shapes(daily, tree):
    d = pd.Timestamp("2026-10-20")
    fcs = forecast_all(daily, d, tree)
    assert set(tree.kpis) <= set(fcs)
    assert "paying_merchant__formula" in fcs
    for f in fcs.values():
        assert f.month_end == "2026-10-31"
        assert f.days_remaining == 11
        assert len(f.path) == 11
        assert pd.notna(f.value)


def test_accumulate_forecast_at_least_mtd(daily, tree):
    fcs = forecast_all(daily, pd.Timestamp("2026-09-10"), tree)
    for k in ("new_merchant", "expiry_merchant", "renewed_merchant"):
        assert fcs[k].value >= fcs[k].to_date


def test_forecast_on_month_end_equals_actual(daily, tree):
    me = pd.Timestamp("2026-09-30")
    fcs = forecast_all(daily, me, tree)
    assert fcs["paying_merchant"].value == daily.loc[me, "paying_merchant"]
    assert fcs["new_merchant"].value == daily.loc["2026-09-01":"2026-09-30", "new_merchant"].sum()
    assert fcs["paying_merchant__formula"].value == daily.loc[me, "paying_merchant"]


def test_month_end():
    assert month_end(pd.Timestamp("2024-02-10")) == pd.Timestamp("2024-02-29")
    assert month_end(pd.Timestamp("2024-02-29")) == pd.Timestamp("2024-02-29")


def test_backtest_shape(df, tree):
    bt = backtest(df, verticals=["hotel"], n_months=2, origin_days=(10,), tree=tree)
    n_kpis = len(tree.kpis) + 1  # + formula forecast
    assert len(bt) == 2 * n_kpis
    assert (bt["ape"] >= 0).all()
    s = backtest_summary(bt)
    assert len(s) == n_kpis and {"mape", "mape_naive", "mape_snaive"} <= set(s.columns)
