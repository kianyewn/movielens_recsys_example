# Python cleanup targets
.PHONY: clean clean-pyc clean-test clean-build clean-all

clean-pyc:  ## Remove Python file artifacts
	find . -type f -name '*.pyc' -delete
	find . -type f -name '*.pyo' -delete
	find . -type f -name '*.pyd' -delete
	find . -type f -name '__pycache__' -delete
	find . -type d -name '__pycache__' -exec rm -rf {} +
	find . -type f -name '*.so' -delete

clean-test: ## Remove test and coverage artifacts
	rm -rf .tox/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .coverage.*

clean-build: ## Remove build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .eggs/

clean-logs: ## Remove log files
	find . -type f -name '*.log' -delete

clean-system: ## Remove system generated files
	find . -type f -name '.DS_Store' -delete
	find . -type f -name 'Thumbs.db' -delete   # Windows thumbnail cache

clean-all: clean-pyc clean-test clean-build clean-logs clean-system ## Remove all artifacts

help:
	@echo "Available commands:"
	@echo "make clean-pyc    - Remove Python file artifacts"
	@echo "make clean-test   - Remove test and coverage artifacts"
	@echo "make clean-build  - Remove build artifacts"
	@echo "make clean-logs   - Remove log files"
	@echo "make clean-system - Remove system generated files like .DS_Store"
	@echo "make clean-all    - Remove all artifacts" 
	@echo "make clean-all    - Remove all artifacts" 