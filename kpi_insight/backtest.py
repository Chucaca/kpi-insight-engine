"""Rolling-origin backtest of the month-end forecasts.

    python -m kpi_insight.backtest

For each vertical and each of the last 12 complete months, forecasts are made
from data up to day 7, 14 and 21 of the month and scored against the realised
month-end value. Writes ``reports/backtest.md`` and ``reports/backtest.csv``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .forecast import backtest, backtest_summary
from .synthetic import load

ROOT = Path(__file__).resolve().parent.parent

NAIVE_DESC = {
    "absolute": "last value",
    "rate": "last value",
    "formula": "last value",
    "accumulate": "MTD run-rate",
    "rate_accumulate": "MTD ratio",
}


def summary_markdown(summary) -> str:
    lines = [
        "| KPI | Type | Model MAPE | Naive MAPE (baseline) | Seasonal-naive MAPE | Improvement vs naive | Improvement vs best baseline |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in summary.itertuples():
        kpi = r.kpi.replace("__formula", " (formula: current + new - expiry + renewed)")
        lines.append(
            f"| {kpi} | {r.type} | {r.mape:.2f}% | {r.mape_naive:.2f}% ({NAIVE_DESC[r.type]}) | "
            f"{r.mape_snaive:.2f}% | {r.improvement_vs_naive_pct:+.0f}% | {r.improvement_vs_best_baseline_pct:+.0f}% |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--months", type=int, default=12)
    ap.add_argument("--out", type=Path, default=ROOT / "reports")
    args = ap.parse_args(argv)
    bt = backtest(load(), n_months=args.months)
    summary = backtest_summary(bt)
    args.out.mkdir(parents=True, exist_ok=True)
    bt.to_csv(args.out / "backtest.csv", index=False)
    by_origin = bt.pivot_table(index="kpi", columns="origin_day", values="ape", aggfunc="mean", sort=False)
    md = [
        "# Month-end forecast backtest (synthetic data)", "",
        f"Rolling origin: last {args.months} complete month-ends x 4 verticals x origins on day 7/14/21 "
        f"({len(bt) // bt['kpi'].nunique()} forecasts per KPI). MAPE of the month-end value.", "",
        summary_markdown(summary), "",
        "Seasonal naive = last observed value on the same weekday as the month-end "
        "(identical to the run-rate for flow KPIs).", "",
        "## MAPE by forecast origin (day of month)", "",
        "| KPI | day 7 | day 14 | day 21 |", "|---|---:|---:|---:|",
    ]
    for kpi, r in by_origin.iterrows():
        md.append(f"| {kpi} | {r[7]:.2f}% | {r[14]:.2f}% | {r[21]:.2f}% |")
    (args.out / "backtest.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(summary_markdown(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
