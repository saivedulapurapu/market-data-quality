PY ?= .venv/bin/python
ifeq ($(OS),Windows_NT)
PY = .venv/Scripts/python.exe
endif

.PHONY: setup fetch pipeline benchmark reports all test cov dashboard clean

setup:
	python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

fetch:        ## download raw payloads into data/raw (cached, polite)
	$(PY) -m mdq fetch

pipeline:     ## load store, run checks, score, export Parquet
	$(PY) -m mdq pipeline

benchmark:    ## injected-error benchmark -> reports/BENCHMARK.md
	$(PY) -m mdq benchmark

reports:      ## reports/DATA_PROFILE.md, reports/QUALITY_REPORT.md
	$(PY) -m mdq reports

all: fetch pipeline benchmark reports

test:
	$(PY) -m pytest

cov:
	$(PY) -m pytest --cov=mdq --cov-report=term-missing

dashboard:
	$(PY) -m streamlit run app/streamlit_app.py

clean:
	rm -rf data/processed .pytest_cache .coverage coverage.xml
