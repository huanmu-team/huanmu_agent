.PHONY: all format lint test help

# Default target executed when no arguments are given to make.
all: help

# Define a variable for the test file path.
TEST_FILE ?= src/

test:
	python -m pytest $(TEST_FILE)

######################
# LINTING AND FORMATTING
######################

# Define a variable for Python files
PYTHON_FILES=src/
MYPY_CACHE=.mypy_cache

lint:
	python -m ruff check .
	python -m ruff format $(PYTHON_FILES) --diff
	python -m ruff check --select I $(PYTHON_FILES)
	python -m mypy --strict $(PYTHON_FILES)

format:
	ruff format $(PYTHON_FILES)
	ruff check --select I --fix $(PYTHON_FILES)

######################
# HELP
######################

help:
	@echo '----'
	@echo 'format - run code formatters'
	@echo 'lint   - run linters'
	@echo 'test   - run tests'
