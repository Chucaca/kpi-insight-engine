"""FastAPI service.

    uvicorn kpi_insight.api:app --port 8000

GET  /health
POST /brief  {"vertical": "fnb", "date": "2026-10-28"}
"""
from __future__ import annotations

import datetime as dt
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from . import __version__
from .narrative import generate_brief, resolve_provider
from .pipeline import analyze
from .synthetic import VERTICALS

app = FastAPI(title="KPI Insight Engine", version=__version__,
              description="Automated daily KPI brief on synthetic POS SaaS data.")

Vertical = Literal[tuple(VERTICALS)]  # type: ignore[valid-type]


class BriefRequest(BaseModel):
    vertical: Vertical
    date: dt.date


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__, "llm_provider": resolve_provider() or "template"}


@app.post("/brief")
def brief(req: BriefRequest) -> dict:
    try:
        analysis = analyze(req.vertical, req.date.isoformat())
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc.args[0])) from exc
    result = generate_brief(analysis.facts)
    return {"vertical": req.vertical, "date": req.date.isoformat(), **result.to_dict(), "facts": analysis.facts}
