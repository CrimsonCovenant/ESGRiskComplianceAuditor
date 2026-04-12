.PHONY: install lint test run

install:
	pip install -e ".[dev]"
	pre-commit install

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/
	mypy src/

test:
	pytest tests/ -v

run:
	streamlit run src/esg_auditor/app.py
