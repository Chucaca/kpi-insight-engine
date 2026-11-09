"""Matplotlib charts for the brief (PNG, headless)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import pandas as pd  # noqa: E402

from .trend import fitted_line  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e1"
BLUE = "#2a78d6"     # series / increase
ORANGE = "#eb6834"   # forecast
RED = "#e34948"      # decrease
NEUTRAL = "#9a9993"  # totals

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
})

_thousands = mticker.FuncFormatter(lambda x, _: f"{x:,.0f}")


def kpi_trend_chart(analysis, path: Path, kpi: str | None = None, days: int = 180) -> Path:
    kpi = kpi or analysis.facts["headline_kpi"]
    label = analysis.facts["kpis"][kpi]["label"]
    s = analysis.daily[kpi].iloc[-days:]
    trend = fitted_line(analysis.daily[kpi], window=days)
    fc = analysis.forecasts[kpi]
    fc_path = pd.Series(fc.path).rename(index=pd.Timestamp)

    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=130)
    ax.plot(s.index, s.values, color=BLUE, lw=1.0, alpha=0.45, label="Daily actual")
    ax.plot(s.index, s.rolling(7).mean(), color=BLUE, lw=2, label="7-day mean")
    ax.plot(trend.index, trend.values, color=INK_2, lw=1.5, ls="--", label="Robust trend (RLM, HuberT)")
    if len(fc_path):
        joined = pd.concat([s.iloc[-1:], fc_path])
        ax.plot(joined.index, joined.values, color=ORANGE, lw=2, label="Forecast to month-end")
        ax.scatter([fc_path.index[-1]], [fc_path.iloc[-1]], s=50, color=ORANGE, zorder=5,
                   edgecolor=SURFACE, linewidth=2)
        ax.annotate(f"{fc_path.iloc[-1]:,.0f}", (fc_path.index[-1], fc_path.iloc[-1]),
                    textcoords="offset points", xytext=(6, 6), color=INK, fontsize=9)
    ax.axvline(analysis.date, color=GRID, lw=1)
    ax.yaxis.set_major_formatter(_thousands)
    ax.set_title(f"{label} - {analysis.vertical.upper()}, last {days} days + month-end forecast")
    ax.legend(frameon=False, loc="upper left", fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def waterfall_chart(analysis, path: Path) -> Path:
    wf = analysis.facts["headline_waterfall"]
    labels, bottoms, heights, colors = [], [], [], []
    running = 0.0
    for w in wf:
        lab = w["label"].replace(" (via paying merchants)", "\n(via paying)").replace(" effect", "\neffect")
        lab = lab.replace(" 20", "\n20") if w["kind"] != "step" else lab
        if w["kind"] in ("start", "end"):
            bottoms.append(0.0)
            heights.append(w["value"])
            colors.append(NEUTRAL)
            running = w["value"]
        else:
            bottoms.append(running if w["value"] >= 0 else running + w["value"])
            heights.append(abs(w["value"]))
            colors.append(BLUE if w["value"] >= 0 else RED)
            running += w["value"]
        labels.append(lab)

    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=130)
    x = range(len(wf))
    ax.bar(x, heights, bottom=bottoms, color=colors, width=0.6, edgecolor=SURFACE, linewidth=2)
    for i, w in enumerate(wf):
        txt = f"{w['value']:,}" if w["kind"] != "step" else f"{w['value']:+,}"
        ax.text(i, bottoms[i] + heights[i], txt, ha="center", va="bottom", fontsize=9, color=INK)
    lo = min(b for b, k in zip(bottoms, wf) if k["kind"] == "step")
    hi = max(b + h for b, h in zip(bottoms, heights))
    pad = (hi - lo) * 0.25
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xticks(list(x), labels, fontsize=8)
    ax.yaxis.set_major_formatter(_thousands)
    ax.grid(axis="x", visible=False)
    ax.set_title(f"Why active merchants moved - {analysis.vertical.upper()} "
                 f"{analysis.compare_date.date()} -> {analysis.date.date()} (axis truncated)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def movers_chart(analysis, path: Path, kpi: str | None = None) -> Path:
    kpi = kpi or analysis.facts["headline_kpi"]
    dims = [k.split("|")[1] for k in analysis.movers if k.startswith(kpi + "|")]
    fig, axes = plt.subplots(1, len(dims), figsize=(9, 4.2), dpi=130,
                             gridspec_kw={"width_ratios": [2.2] + [1.6] * (len(dims) - 1)})
    axes = [axes] if len(dims) == 1 else list(axes)
    for ax, dim in zip(axes, dims):
        m = analysis.movers[f"{kpi}|{dim}"]
        m = m[~m["filtered"]].sort_values("contribution")
        colors = [BLUE if c >= 0 else RED for c in m["contribution"]]
        ax.barh(m["member"].astype(str), m["contribution"], color=colors, height=0.6,
                edgecolor=SURFACE, linewidth=2)
        for yi, c in enumerate(m["contribution"]):
            ax.text(c, yi, f" {c:+,.0f} ", va="center", ha="left" if c >= 0 else "right", fontsize=8, color=INK)
        ax.axvline(0, color=INK_2, lw=0.8)
        lim = m["contribution"].abs().max() * 1.45
        ax.set_xlim(-lim, lim)
        ax.xaxis.set_major_formatter(_thousands)
        ax.grid(axis="y", visible=False)
        ax.set_title(f"by {dim}", fontsize=10)
        ax.tick_params(axis="y", labelsize=8)
    fig.suptitle(f"Contribution to the change in {analysis.facts['kpis'][kpi]['label'].lower()} "
                 f"({analysis.vertical.upper()}, vs same day last month)",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def render_all(analysis, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    v = analysis.vertical
    return {
        "trend": kpi_trend_chart(analysis, out_dir / f"{v}_kpi_trend.png"),
        "waterfall": waterfall_chart(analysis, out_dir / f"{v}_waterfall.png"),
        "movers": movers_chart(analysis, out_dir / f"{v}_movers.png"),
    }
