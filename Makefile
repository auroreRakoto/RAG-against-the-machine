MANAGER     = uv
EXEC        = python3
# PACKAGE     = src

VENV_DIR    = .venv

UV_CACHE    = .cache/uv-cache

install:
	$(MANAGER) sync

run:
	clear
	$(MANAGER) run $(EXEC) main.py

debug:
	$(MANAGER) run $(EXEC) -m pdb -m $(PACKAGE)

clean:
	rm -rf $(VENV_DIR)
	rm -rf $(UV_CACHE)
	rm -rf .pytest_cache
	rm -rf __pycache__

lint:
	$(MANAGER) run flake8 .
	$(MANAGER) run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

.PHONY: install run debug clean lint