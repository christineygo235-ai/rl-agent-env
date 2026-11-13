# All targets are local and offline (after the one-time `make venv`).
PYTHON ?= python3
VENV   := .venv
BIN    := $(VENV)/bin

.PHONY: help venv test test-all tasks reference-check redteam calibrate evidence clean

help:
	@echo "make venv   - create .venv and install rlenv + dev deps (network once)"
	@echo "make test   - run the fast offline test suite (skips tests marked slow)"
	@echo "make test-all - all offline tests incl. slow end-to-end runs (~1 min)"
	@echo "make tasks  - validate and list all task specs"
	@echo "make reference-check - full reference runs vs thresholds on held-out seeds (slow)"
	@echo "make redteam - reference + floor + adversarial agents vs expected rewards (~1 min)"
	@echo "make calibrate - reference over 10 seeds per task -> calibration/ (several minutes)"
	@echo "make evidence - regenerate docs/EVIDENCE.md from committed real artifacts"

$(BIN)/rlenv: pyproject.toml
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --quiet --upgrade pip
	$(BIN)/pip install --quiet -e '.[dev]'
	@touch $@

venv: $(BIN)/rlenv

test: venv
	$(BIN)/pytest -m "not slow"

test-all: venv
	$(BIN)/pytest

tasks: venv
	$(BIN)/rlenv list-tasks

reference-check: venv
	$(BIN)/rlenv check-reference all

calibrate: venv
	$(BIN)/rlenv calibrate all --seeds 10

redteam: venv
	$(BIN)/rlenv redteam frozenlake-tabular

evidence: venv
	$(BIN)/rlenv evidence

clean:
	rm -rf $(VENV) .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
