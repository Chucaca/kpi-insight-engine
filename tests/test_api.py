from fastapi.testclient import TestClient

from kpi_insight.api import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["llm_provider"] == "template"


def test_brief_endpoint():
    r = client.post("/brief", json={"vertical": "hotel", "date": "2026-10-28"})
    assert r.status_code == 200
    body = r.json()
    assert body["source"] == "template"
    assert set(body["brief"]) == {"title", "summary", "drivers", "recommendations"}
    assert body["facts"]["vertical"] == "hotel"


def test_brief_validation():
    assert client.post("/brief", json={"vertical": "casino", "date": "2026-10-28"}).status_code == 422
    assert client.post("/brief", json={"vertical": "fnb", "date": "2030-01-01"}).status_code == 404
