PY ?= python

.PHONY: test test-fast lint fix

test:  ## all tests (CPU, about 40 min)
	$(PY) -m pytest tests

test-fast:  ## without the tests marked slow (about 10 min)
	$(PY) -m pytest tests -m "not slow"

lint:  ## ruff, as in CI
	ruff check .

fix:  ## ruff's safe fixes
	ruff check --fix .
