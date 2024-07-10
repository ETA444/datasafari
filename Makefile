.PHONY: help clean clean-build clean-pyc clean-test lint test coverage docs release dist install clean-docs servedocs activate

help:
	@echo "clean        - remove all build, test, coverage, and Python artifacts"
	@echo "clean-build  - remove build artifacts"
	@echo "clean-pyc    - remove Python file artifacts"
	@echo "clean-test   - remove test and coverage artifacts"
	@echo "lint         - check style with flake8"
	@echo "test         - run tests quickly with the default Python"
	@echo "coverage     - check code coverage quickly with the default Python"
	@echo "docs         - generate Sphinx HTML documentation, including API docs"
	@echo "clean-docs   - remove previously built docs"
	@echo "servedocs    - compile the docs watching for changes"
	@echo "release      - package and upload a release"
	@echo "dist         - builds source and wheel package"
	@echo "install      - install the package to the active Python's site-packages"
	@echo "activate     - activate the virtual environment"

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage, and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

lint: ## check style with flake8
	poetry run flake8 datasafari tests

test: ## run tests quickly with the default Python
	poetry run pytest

coverage: ## check code coverage quickly with the default Python
	poetry run coverage run --source datasafari -m pytest
	poetry run coverage report -m
	poetry run coverage html
	@echo "Open htmlcov/index.html in your browser to view the report."

docs: clean-docs ## generate Sphinx HTML documentation, including API docs
	poetry run sphinx-build -b html docs/ docs/_build/html
	@echo "Open docs/_build/html/index.html in your browser to view the documentation."

clean-docs: ## remove previously built docs
	rm -rf docs/_build/

servedocs: docs ## compile the docs watching for changes
	poetry run watchmedo shell-command -p '*.rst' -c 'make docs' -R -D .

release: dist ## package and upload a release
	poetry publish --username ETA444

dist: clean ## builds source and wheel package
	poetry build

install: clean ## install the package to the active Python's site-packages
	poetry install

activate: ## activate the virtual environment
	source venv/bin/activate && exec $$SHELL
