.PHONY: install test lint format check-format check all clean

# Install development dependencies
install:
	pip install -e .[dev]

# Run tests
test:
	pytest --tb=short -v

# Lint code with ruff
lint:
	ruff check src/ tests/

# Format code with black
format:
	black src/ tests/

# Check formatting without modifying
check-format:
	black --check src/ tests/

# Run all checks (mimics CI)
check: lint check-format test

# Run all checks and fix issues
all: format lint test

# Clean cache files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete