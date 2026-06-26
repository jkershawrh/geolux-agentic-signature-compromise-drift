.PHONY: test test-unit test-integration test-property test-contract test-bdd install lint

install:
	pip install -e ".[dev]"

test:
	python -m pytest tests/ -q --ignore=tests/bdd
	python -m behave tests/bdd/features/ --no-capture

test-unit:
	python -m pytest tests/unit/ -v

test-integration:
	python -m pytest tests/integration/ -v -m integration

test-property:
	python -m pytest tests/property/ -v

test-contract:
	python -m pytest tests/contract/ -v

test-bdd:
	python -m behave tests/bdd/features/ --no-capture

lint:
	python -m ruff check .
	python -m ruff format --check .

rubrics:
	python scripts/evaluate_rubrics.py
