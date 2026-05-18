.PHONY: install lint format test run docker-up docker-down ingest clean

install:
	pip install -e ".[dev]"

lint:
	ruff check . && mypy .

format:
	ruff format .

test:
	pytest tests/ -v --cov=. --cov-report=term-missing

run:
	streamlit run app.py

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

ingest:
	python -m cli ingest --pdf-dir data/pdfs --collection hybrid_bench

clean:
	find . -type d -name __pycache__ | xargs rm -rf
