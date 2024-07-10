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
	flake8 datasafari tests

test: ## run tests quickly with the default Python
	pytest

coverage: ## check code coverage quickly with the default Python
	coverage run --source datasafari -m pytest
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html

docs: clean-docs ## generate Sphinx HTML documentation, including API docs
	sphinx-apidoc -o docs/source/ datasafari
	$(MAKE) -C docs html
	$(BROWSER) docs/_build/html/index.html

clean-docs: ## remove previously built docs
	rm -rf docs/_build/
	rm -f docs/datasafari.rst
	rm -f docs/modules.rst

servedocs: docs ## compile the docs watching for changes
	watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs html' -R -D .

release: dist ## package and upload a release
	twine upload dist/*

dist: clean ## builds source and wheel package
	python setup.py sdist
	python setup.py bdist_wheel
	ls -l dist

install: clean ## install the package to the active Python's site-packages
	python setup.py install
