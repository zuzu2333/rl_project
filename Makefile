# These have been configured to only really run short tasks. Longer form tasks
# are usually completed in github actions.

NAME := RL_project
PACKAGE_NAME := rl_project_mb_vs_mf

DIR := "${CURDIR}"
SOURCE_DIR := ${PACKAGE_NAME}
DIST := dist
TESTS_DIR := tests

.PHONY: help install check format pre-commit clean clean-build build publish test

help:
	@echo "Makefile ${NAME}"
	@echo "* install      	  to install all equirements and install pre-commit"
	@echo "* clean            to clean any doc or build files"
	@echo "* check            to check the source code for issues"
	@echo "* format           to format the code with black and isort"
	@echo "* pre-commit       to run the pre-commit check"
	@echo "* build            to build a dist"
	@echo "* publish          to help publish the current branch to pypi"
	@echo "* test             to run the tests"

PYTHON ?= python
PYTEST ?= uv run pytest
PIP ?= uv pip
MAKE ?= make
PRECOMMIT ?= uv run pre-commit
RUFF ?= uv run ruff

install:
	$(PIP) install swig
	$(PIP) install -e ".[dev]"
	pre-commit install

check: 
	$(RUFF) format --check rl_exercises tests
	$(RUFF) check rl_exercises tests

pre-commit:
	$(PRECOMMIT) run --all-files

format: 
	uv run isort rl_exercises tests
	$(RUFF) format --silent rl_exercises tests
	$(RUFF) check --fix --silent rl_exercises tests --exit-zero
	$(RUFF) check --fix rl_exercises tests --exit-zero



clean-build:
	$(PYTHON) setup.py clean
	rm -rf ${DIST}

# Build a distribution in ./dist
build:
	uv build

# Clean up any builds in ./dist as well as doc, if present
clean: clean-build 