# Development Testing Guide

## Running Tests

### Quick Test Suite
Run the core correctness and duplication detection tests:

```bash
pytest tests/test_deterministic_mode.py tests/test_duplication_detection.py -v
```

### Full Test Suite
Run all tests (excluding GPU-specific tests if no GPU available):

```bash
pytest tests/ -k "not gpu" -v
```

### Specific Test Categories

**Deterministic Mode Tests:**
```bash
pytest tests/test_deterministic_mode.py -v
```

**Duplication Detection Tests:**
```bash
pytest tests/test_duplication_detection.py -v
```

**CPU Correctness Tests:**
```bash
pytest tests/test_specdec_cpu_correctness.py -v
```

## Test Requirements

All tests require:
- Python 3.9+
- `torch>=2.0.0`
- `transformers>=4.30.0`
- `pytest>=7.4.0`

Install dependencies:
```bash
pip install -r requirements.txt
```

## CI/CD

The CI pipeline runs:
1. **Lint**: Code formatting (black, isort), linting (flake8), type checking (mypy)
2. **Test CPU**: Unit tests on CPU
3. **Test Integration**: Integration tests (if available)

To run locally:
```bash
# Linting
black --check src/ tests/ scripts/
isort --check-only src/ tests/ scripts/
flake8 src/ tests/ scripts/

# Tests
pytest tests/ -k "not gpu" -v
```





