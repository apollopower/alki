# Contributing to Alki 🌊

Thank you for your interest in contributing to **Alki**, an open-source tool for deploying LLMs at the edge. This project is at an early stage, and contributions of all kinds are welcome — from code and documentation to testing and discussions.

## 📋 Code of Conduct

By participating, you agree to uphold the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). Be respectful and constructive.

## 🏁 Getting Started

### Prerequisites

* Python 3.10+
* `pipx` or `poetry/pdm` (preferred) for dependency management
* (Optional) Docker if working on backend builder images

### Repository Setup

```bash
# Setup your venv
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# or .venv\Scripts\activate on Windows

# Install in editable mode
pip install -e .[dev]
```

### Development Workflow

**IMPORTANT**: Always activate the virtual environment before running any commands:
```bash
source .venv/bin/activate  # Required for all commands below
```

```bash
# Setup environment
make install

# Before committing/pushing, run all checks
make check

# Auto-format code and run checks
make all

# Individual tasks
make test          # Run tests only
make format        # Format code with black
make lint          # Lint with ruff
make clean         # Clean cache files

# Test real ONNX export with actual models (optional, not part of CI)
python scripts/test_onnx_export_e2e.py --model distilgpt2

# Test end-to-end quantization pipeline (optional, takes ~2 minutes)
python scripts/test_quantization_e2e.py
```

**Pre-push checklist:**
- Ensure virtual environment is activated (`source .venv/bin/activate`)
- Run `make check` to ensure all CI checks pass locally
- This runs the same linting, formatting, and test checks as GitHub Actions
- Optionally run real integration tests for end-to-end validation

## 📦 Project Structure (early draft)

```
alki/
  cli/              # CLI commands (Typer)
  core/             # ingest, quantize, package
  backends/         # backend plugins (ORT GenAI, OpenVINO, etc.)
  runtime/          # runners, kv-cache helpers
  validate/         # accuracy & perf harness
  presets/          # built-in configs
  examples/         # model recipes
docker/             # optional builder Dockerfiles
tests/              # pytest suite
```

## 🚀 Ways to Contribute

* **Code**: new features, bug fixes, optimization passes, backend support
* **Docs**: tutorials, clarifications, diagrams
* **Testing**: run models on your hardware (Jetson, Intel, Apple Silicon, etc.) and report results
* **Research**: experiment with quantization/pruning and share findings
* **Community**: help with issues, triage, and design discussions

## 🔌 Backend Plugins

Each backend (e.g. `ort_genai`, `trt_llm`, `mlx_exec`) is a self-contained module that exposes:

```python
def prepare(model_ir, config) -> Bundle:
    """Compile/quantize/convert the model and return a Bundle."""
```

This makes it easy to add new hardware targets without touching the core framework.

## 🧪 Validation

All contributions should include:

* **Unit tests (pytest)** for core logic
* **Integration test** for at least one model (e.g., GPT-2 with W8A8 quant on CPU)
* **Benchmark logs** (optional, but encouraged for backend contributions)

### Quantization Testing

The project includes SmoothQuant W8A8 quantization with comprehensive testing:

* **Unit tests**: `pytest tests/test_quantizer.py` (fast, <5 seconds)
* **End-to-end test**: `python scripts/test_quantization_e2e.py` (slower, ~2 minutes)
* **Quick demo**: `python scripts/demo_quantization.py` (no model download needed)

When contributing quantization-related changes:
- Ensure unit tests pass for configuration validation and smoothing calculations
- Run the end-to-end test to verify compatibility with real models
- Test with different alpha values (0.0, 0.5, 1.0) to verify smoothing behavior

## 📜 License

By contributing, you agree that your contributions will be licensed under the [Apache-2.0 License](LICENSE).

## 💬 Communication

* Use GitHub Issues for bugs/feature requests
* Pull Requests for contributions
* (Future) Discussions forum/Discord for community chatter