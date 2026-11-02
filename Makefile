PY ?= python

.PHONY: test test-fast lint fix

test:  ## all tests (on CPU about 2 min; the simulator's tests need a GPU and Isaac Lab)
	$(PY) -m pytest tests

test-fast:  ## without the tests marked slow (on CPU about 1.5 min)
	$(PY) -m pytest tests -m "not slow"

lint:  ## ruff, as in CI
	ruff check .

fix:  ## ruff's safe fixes
	ruff check --fix .
