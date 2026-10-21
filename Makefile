.PHONY: check

# The full gate: frozen dependencies, Ruff, mypy, tests, example config.
check:
	./scripts/check
