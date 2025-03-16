.PHONY: install test clean examples

# Install the package in development mode
install:
	pip install -e .

# Run tests
test:
	python -m unittest discover tests

# Clean build artifacts
clean:
	rm -rf build/ dist/ *.egg-info/
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	find . -name "*.pyd" -delete
	find . -name ".pytest_cache" -type d -exec rm -rf {} +
	find . -name ".coverage" -delete
	find . -name "htmlcov" -type d -exec rm -rf {} +

# Build distribution package
dist: clean
	python -m build

# Run a specific example
example-boston:
	python examples/BostonData.py

example-survival:
	python examples/survival_analysis_example.py

example-basic:
	python examples/basic/simple_imputation.py