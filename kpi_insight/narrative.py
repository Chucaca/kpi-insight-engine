"""Narrative layer: computed facts -> ``KpiBrief``.

* ``template_brief``: deterministic, offline, always available.
* ``generate_brief``: optional LLM (Gemini via ``google-genai`` or OpenAI via
  ``openai``, through ``instructor`` when installed). The provider is picked by
  ``KPI_LLM_PROVIDER`` (``auto`` | ``gemini`` | ``openai`` | ``none``); ``auto``
  uses whichever API key is set and falls back to the template when none is.
  All LLM SDK imports are lazy, so the package has no hard LLM dependency.
* Guardrail: every number in the LLM text must match a number in the facts
  (with rounding tolerance). Otherwise the LLM is asked again with feedback,
  and after ``max_attempts`` the deterministic template is used.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


class KpiBrief(BaseModel):
    title: str = Field(description="One-line headline for the day")
    summary: str = Field(description="2-4 sentences: what moved, by how much, and the month-end outlook")
    drivers: list[str] = Field(description="Root causes, largest first, each quantified")
    recommendations: list[str] = Field(description="Concrete follow-ups for the business owner")

    def text(self) -> str:
        return "\n".join([self.title, self.summary, *self.drivers, *self.recommendations])


# ------------------------------------------------------------------ guardrail
_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_NUM_RE = re.compile(r"(?<![\w.])[-+−]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:\s?[kKmM]\b)?")
SMALL_INT_WHITELIST = set(range(0, 11))  # "top 3", "2 channels", ...


def _collect_numbers(obj: Any, out: list[float]) -> list[float]:
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect_numbers(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _collect_numbers(v, out)
    return out


def extract_numbers(text: str) -> list[tuple[str, float, float]]:
    """Return ``(raw, value, tolerance)`` for every number in ``text`` (ISO dates skipped).

    The tolerance is half a unit of the last printed digit: "73.3" -> 0.05,
    "1.3k" -> 50, "57k" -> 500, "57,400" -> 50 (trailing zeros = rounding).
    """
    text = _DATE_RE.sub(" ", text)
    found = []
    for m in _NUM_RE.finditer(text):
        raw = m.group(0).strip()
        s = raw.replace(",", "").replace("−", "-").replace(" ", "")
        mult = 1.0
        if s[-1] in "kKmM":
            mult = 1e3 if s[-1] in "kK" else 1e6
            s = s[:-1]
        if "." in s:
            unit = 10.0 ** -len(s.split(".")[1])
        else:
            digits = s.lstrip("+-")
            unit = 10.0 ** (len(digits) - len(digits.rstrip("0"))) if digits.strip("0") else 1.0
        found.append((raw, float(s) * mult, 0.5 * unit * mult))
    return found


def unsupported_numbers(text: str, facts: dict) -> list[str]:
    """Numbers in ``text`` that cannot be traced back to ``facts``.

    A number ``n`` is supported if some fact value ``v`` satisfies
    ``|abs(n) - abs(v)| <= tolerance`` where the tolerance is only the rounding
    implied by how ``n`` is printed (see ``extract_numbers``). Signs are
    ignored ("down 958" == -958). ISO dates and integers 0-10 ("top 3") are allowed.
    """
    allowed = [abs(v) for v in _collect_numbers(facts, [])]
    bad = []
    for raw, n, tol in extract_numbers(text):
        n = abs(n)
        if n in SMALL_INT_WHITELIST and float(n).is_integer() and "." not in raw:
            continue
        if not any(abs(n - v) <= tol + 1e-9 for v in allowed):
            bad.append(raw)
    return bad


# ------------------------------------------------------------------ template
def _i(x: int | float) -> str:
    return f"{int(x):,}"


def _s(x: int | float) -> str:
    return f"{int(x):+,}"


def _p(x: float, unit: str = "%") -> str:
    return f"{x:+.1f}{unit}"


def _net_share_meaningful(m: dict) -> bool:
    """Net shares are only readable when the net change is a sizeable part of gross movement."""
    return m["gross_movement"] and abs(m["total_change"]) >= 0.25 * m["gross_movement"]


def template_brief(facts: dict) -> KpiBrief:
    """Deterministic brief built only from ``facts`` (passes the guardrail by construction)."""
    k = facts["kpis"]
    hk = facts["headline_kpi"]
    h = k[hk]
    rate = k.get("active_rate")
    v = facts["vertical"].upper()
    direction = "down" if h["change"] < 0 else "up"
    title = f"{v} {facts['date']}: {h['label'].lower()} {direction} {_i(abs(h['change']))} ({_p(h['change_pct'])}) vs same day last month"

    t = h.get("trend", {})
    summary = (
        f"{h['label']} stand at {_i(h['current'])}, {direction} {_i(abs(h['change']))} ({_p(h['change_pct'])}) "
        f"versus the same day last month ({facts['compare_date']}). "
    )
    if rate:
        summary += (f"{rate['label']} is {rate['current']:.1f}% ({_p(rate['change'], ' pp')}), while "
                    f"{k['paying_merchant']['label'].lower()} moved {_s(k['paying_merchant']['change'])}. ")
    if t:
        summary += (f"The {t['window_days']}-day robust trend is {t['direction']} ({t['strength']}, "
                    f"{_p(t['change_over_window_pct'])} over the window) and momentum is {t['momentum']}. ")
    summary += (f"Month-end forecast ({facts['month_end']}, a {facts['month_end_weekday']}): "
                f"{_i(h['month_end_forecast'])} {h['label'].lower()}.")

    drivers = []
    steps = sorted([w for w in facts["headline_waterfall"] if w["kind"] == "step"],
                   key=lambda w: abs(w["value"]), reverse=True)
    for w in steps:
        drivers.append(f"{w['label']}: {_s(w['value'])} {h['label'].lower()}.")
    for key, m in facts["movers"].items():
        if not m["top"]:
            continue
        top = m["top"][0]
        kpi_label = k[m["kpi"]]["label"]
        pp = m["unit"] == "pp"
        total = f"{m['total_change']:+.2f} pp" if pp else _s(m["total_change"])
        if _net_share_meaningful(m):
            share = f"{top['share_of_change_pct']:.1f}% of the net {total} change"
        else:
            share = f"{top['share_of_gross_movement_pct']:.1f}% of gross movement; net change only {total}"
        if pp:
            drivers.append(f"{kpi_label} by {m['dimension']}: {top['member']} {top['contribution']:+.2f} pp "
                           f"({share}); its own rate {top['value_prev']:.1f}% -> {top['value_curr']:.1f}%.")
        else:
            basis = " month-to-date" if k[m["kpi"]]["basis"] == "month-to-date" else ""
            drivers.append(f"{kpi_label}{basis} by {m['dimension']}: {top['member']} {_s(top['contribution'])} "
                           f"({share}).")

    recs = []
    prov = facts["movers"].get(f"{hk}|province", {}).get("top", [])
    if prov and prov[0]["share_of_gross_movement_pct"] >= 40 and "active_rate_effect" in prov[0]:
        p = prov[0]
        recs.append(
            f"Investigate engagement in {p['member']}: its active rate went from {p['rate_prev_pct']:.1f}% to "
            f"{p['rate_curr_pct']:.1f}% (effect {_s(p['active_rate_effect'])} active merchants). Check incidents, "
            f"support tickets and field-team coverage there.")
    for key, m in facts["movers"].items():
        if m["kpi"] != "new_merchant" or not m["top"]:
            continue
        top = m["top"][0]
        if top["contribution"] < 0 and top["share_of_gross_movement_pct"] >= 40 and top["value_prev"]:
            recs.append(
                f"Review acquisition in the {top['member']} {m['dimension']}: month-to-date new merchants "
                f"{_i(top['value_prev'])} -> {_i(top['value_curr'])}. Confirm pipeline and partner activity.")
    exp = k.get("expiry_merchant")
    if exp and exp["change_pct"] > 5:
        recs.append(f"Expiries are {_p(exp['change_pct'])} month-to-date; prioritise renewal outreach.")
    if not recs:
        recs.append("No single segment dominates the change; keep monitoring and re-check tomorrow.")
    return KpiBrief(title=title, summary=summary.strip(), drivers=drivers, recommendations=recs)


# ------------------------------------------------------------------ LLM layer
SYSTEM_PROMPT = """You are a senior analyst writing a daily KPI brief for business leaders of a
multi-vertical POS SaaS. Use ONLY the numbers in FACTS (you may round them, never invent,
add, subtract or recompute numbers). Explain what moved, why (use the waterfall and the movers),
and the month-end outlook. Be concise and concrete. Return JSON matching the schema:
title (str), summary (str, 2-4 sentences), drivers (list of str), recommendations (list of str)."""


def build_prompt(facts: dict, feedback: str | None = None) -> str:
    p = f"FACTS:\n{json.dumps(facts, indent=1, ensure_ascii=False)}"
    if feedback:
        p += f"\n\nYOUR PREVIOUS ANSWER WAS REJECTED: {feedback}"
    return p


def resolve_provider() -> str | None:
    choice = os.getenv("KPI_LLM_PROVIDER", "auto").lower()
    if choice in ("none", "template", "off"):
        return None
    has_gemini = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    if choice == "gemini":
        return "gemini" if has_gemini else None
    if choice == "openai":
        return "openai" if has_openai else None
    return "gemini" if has_gemini else "openai" if has_openai else None


def _call_gemini(prompt: str) -> KpiBrief:
    from google import genai  # lazy
    from google.genai import types

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    resp = client.models.generate_content(
        model=os.getenv("KPI_LLM_MODEL", "gemini-2.5-flash"),
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=KpiBrief,
            temperature=0.2,
        ),
    )
    return resp.parsed if isinstance(resp.parsed, KpiBrief) else KpiBrief.model_validate_json(resp.text)


def _call_openai(prompt: str) -> KpiBrief:
    from openai import OpenAI  # lazy

    model = os.getenv("KPI_LLM_MODEL", "gpt-4o-mini")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    try:
        import instructor  # optional, lazy
    except ImportError:
        instructor = None
    if instructor is not None:
        client = instructor.from_openai(OpenAI())
        return client.chat.completions.create(model=model, response_model=KpiBrief, messages=messages,
                                              temperature=0.2, max_retries=1)
    resp = OpenAI().chat.completions.create(model=model, messages=messages, temperature=0.2,
                                            response_format={"type": "json_object"})
    return KpiBrief.model_validate_json(resp.choices[0].message.content)


PROVIDERS: dict[str, Callable[[str], KpiBrief]] = {"gemini": _call_gemini, "openai": _call_openai}


@dataclass
class BriefResult:
    brief: KpiBrief
    source: str                      # "template" | "llm:<provider>" | "template_fallback"
    attempts: int = 0
    rejected: list[list[str]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"brief": self.brief.model_dump(), "source": self.source,
                "attempts": self.attempts, "rejected_numbers": self.rejected}


def generate_brief(facts: dict, llm: Callable[[str], Any] | None = None, provider: str | None = None,
                   max_attempts: int = 2) -> BriefResult:
    """Produce a brief. ``llm`` (prompt -> KpiBrief | dict | JSON str) overrides provider lookup."""
    name = "custom"
    if llm is None:
        provider = provider or resolve_provider()
        if provider is None:
            return BriefResult(template_brief(facts), "template")
        llm, name = PROVIDERS[provider], provider

    rejected: list[list[str]] = []
    feedback = None
    for attempt in range(1, max_attempts + 1):
        try:
            out = llm(build_prompt(facts, feedback))
            brief = (out if isinstance(out, KpiBrief) else
                     KpiBrief.model_validate_json(out) if isinstance(out, str) else KpiBrief.model_validate(out))
        except Exception as exc:  # network, quota, schema errors -> fall back
            log.warning("LLM call failed (%s): %s", name, exc)
            rejected.append([f"error: {type(exc).__name__}"])
            continue
        bad = unsupported_numbers(brief.text(), facts)
        if not bad:
            return BriefResult(brief, f"llm:{name}", attempt, rejected)
        log.warning("guardrail rejected numbers %s (attempt %d)", bad, attempt)
        rejected.append(bad)
        feedback = (f"these numbers do not appear in FACTS: {', '.join(bad)}. "
                    "Rewrite using only numbers present in FACTS.")
    return BriefResult(template_brief(facts), "template_fallback", max_attempts, rejected)
