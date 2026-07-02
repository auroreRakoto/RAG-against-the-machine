PYTHON ?= python3
VENV_DIR ?= .venv

ifeq ($(OS),Windows_NT)
VENV_PYTHON := $(VENV_DIR)/Scripts/python.exe
else
VENV_PYTHON := $(VENV_DIR)/bin/python
endif

install:
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install -e .
	$(VENV_PYTHON) -m pip install pytest

run:
	$(VENV_PYTHON) -m src

test:
	$(VENV_PYTHON) -m pytest -q

test-retrieval:
	$(VENV_PYTHON) -m pytest -q tests/test_retrieval.py

debug:
	$(VENV_PYTHON) -m pdb -m src

clean:
	rm -rf $(VENV_DIR)
	rm -rf .pytest_cache
	rm -rf __pycache__
	rm -rf **/__pycache__

lint:
	$(VENV_PYTHON) -m pip install flake8 mypy
	$(VENV_PYTHON) -m flake8 .
	$(VENV_PYTHON) -m mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

.PHONY: install run test test-retrieval debug clean lint