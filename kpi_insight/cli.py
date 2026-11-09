"""Command line entry point.

    python -m kpi_insight.cli --vertical fnb --date 2026-10-28

Writes ``reports/brief_<vertical>_<date>.md`` (+ ``.json`` facts) and PNG
charts to ``docs/img/``.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .charts import render_all
from .narrative import generate_brief
from .pipeline import analyze
from .report import render_markdown
from .synthetic import END, VERTICALS

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATE = "2026-10-28"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate an automated daily KPI brief (synthetic data).")
    ap.add_argument("--vertical", default="fnb", choices=VERTICALS)
    ap.add_argument("--date", default=DEFAULT_DATE, help=f"YYYY-MM-DD (data ends {END})")
    ap.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    ap.add_argument("--img-dir", type=Path, default=ROOT / "docs" / "img")
    ap.add_argument("--no-charts", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    analysis = analyze(args.vertical, args.date)
    result = generate_brief(analysis.facts)
    images = None if args.no_charts else render_all(analysis, args.img_dir)

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    stem = f"brief_{args.vertical}_{analysis.date.date()}"
    md_path = args.reports_dir / f"{stem}.md"
    md_path.write_text(render_markdown(analysis.facts, result, images, args.reports_dir), encoding="utf-8")
    (args.reports_dir / f"{stem}.json").write_text(
        json.dumps({"facts": analysis.facts, **result.to_dict()}, indent=2), encoding="utf-8")
    print(f"brief  -> {md_path}  (narrative: {result.source})")
    for p in (images or {}).values():
        print(f"chart  -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
