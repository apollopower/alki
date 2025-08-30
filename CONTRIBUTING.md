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
# Install in editable mode
pip install -e .[dev]
```

### Running Tests

```bash
pytest tests/
```

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

* Unit tests (pytest) for core logic
* Integration test for at least one model (e.g., Llama-3.2-3B with W8A8 quant on CPU)
* Benchmark logs (optional, but encouraged for backend contributions)

## 📜 License

By contributing, you agree that your contributions will be licensed under the [Apache-2.0 License](LICENSE).

## 💬 Communication

* Use GitHub Issues for bugs/feature requests
* Pull Requests for contributions
* (Future) Discussions forum/Discord for community chatter