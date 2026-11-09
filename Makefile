PY ?= python
VERTICAL ?= fnb
DATE ?= 2026-10-28

.PHONY: install test brief backtest serve docker-build docker-run

install:
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest -q

brief:
	$(PY) -m kpi_insight.cli --vertical $(VERTICAL) --date $(DATE)

backtest:
	$(PY) -m kpi_insight.backtest

serve:
	$(PY) -m uvicorn kpi_insight.api:app --reload --port 8000

docker-build:
	docker build -t kpi-insight-engine .

docker-run:
	docker run --rm -p 8000:8000 -e GEMINI_API_KEY -e OPENAI_API_KEY kpi-insight-engine
