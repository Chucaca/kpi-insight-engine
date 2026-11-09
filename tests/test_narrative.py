import pytest

from kpi_insight.narrative import KpiBrief, generate_brief, template_brief, unsupported_numbers
from kpi_insight.pipeline import analyze

FACTS = {"kpis": {"active_merchant": {"current": 57009, "change": -1304, "change_pct": -2.2},
                  "active_rate": {"current": 73.3, "change": -1.6}}}


@pytest.fixture(scope="module")
def facts(df):
    return analyze("fnb", "2026-10-28", df=df).facts


def test_guardrail_accepts_numbers_from_facts():
    text = ("Active merchants are 57,009 (down 1,304, -2.2%) on 2026-10-28; "
            "active rate 73.3% (-1.6 pp). Top 3 movers.")
    assert unsupported_numbers(text, FACTS) == []


def test_guardrail_accepts_rounding():
    assert unsupported_numbers("about 57k active merchants, rate ~73%, down 1.3k", FACTS) == []


def test_guardrail_rejects_hallucinated_number():
    text = "Active merchants are 57,009, down 1,304, and 48,213 merchants churned in Harborview (12.7%)."
    assert unsupported_numbers(text, FACTS) == ["48,213", "12.7"]


def test_template_passes_guardrail(facts):
    b = template_brief(facts)
    assert unsupported_numbers(b.text(), facts) == []
    assert "Harborview" in b.text()


def test_no_key_uses_template(facts):
    assert generate_brief(facts).source == "template"


def test_llm_hallucination_falls_back_to_template(facts):
    calls = []

    def fake_llm(prompt):
        calls.append(prompt)
        return KpiBrief(title="Bad day", summary="Active merchants fell by 99,999.",
                        drivers=["Harborview lost 31,337 merchants"], recommendations=["Panic"])

    r = generate_brief(facts, llm=fake_llm, max_attempts=2)
    assert r.source == "template_fallback"
    assert len(calls) == 2 and "REJECTED" in calls[1] and "99,999" in calls[1]
    assert r.brief == template_brief(facts)


def test_llm_retry_then_success(facts):
    good = template_brief(facts)
    answers = iter([
        {"title": "x", "summary": "Down 77,777.", "drivers": [], "recommendations": []},
        good.model_dump_json(),
    ])
    r = generate_brief(facts, llm=lambda p: next(answers), max_attempts=3)
    assert r.source == "llm:custom" and r.attempts == 2 and r.rejected == [["77,777"]]


def test_llm_error_falls_back(facts):
    def broken(prompt):
        raise TimeoutError("network")

    assert generate_brief(facts, llm=broken).source == "template_fallback"
