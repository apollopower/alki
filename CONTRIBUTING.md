# Contributing to Alki 🌊

Thank you for your interest in contributing to **Alki**, an open-source toolchain for deploying LLMs at the edge with llama.cpp. This project is focused on production-ready edge deployments, and contributions of all kinds are welcome — from code and documentation to testing and discussions.

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

# Test end-to-end packing pipeline with real models (optional, not part of CI)
python scripts/test_pack_e2e.py

# Test with different models and quantization profiles
python scripts/test_pack_e2e.py --model Qwen/Qwen3-0.6B-Instruct --quant Q4_K_M,Q5_K_M

# Test container image generation
python scripts/test_image_build.py

# Test runtime inference with created bundles
llama-server -m ./dist/qwen3-0.6b/models/qwen3-0_6b-instruct-q4_k_m.gguf --api --host 0.0.0.0 --port 8080
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
  core/             # HF→GGUF conversion, bundle packaging
  backends/         # runtime plugins (llama.cpp, Ollama, MLC, ONNX)
  converters/       # GGUF conversion and quantization
  deploy/           # deployment configuration generators
  validate/         # bundle validation & smoke tests
  examples/         # model examples and deployment configs
docker/             # container templates and Dockerfiles
tests/              # pytest suite
```

## 🚀 Ways to Contribute

Alki is focused on production-ready LLM deployment at the edge with llama.cpp. Contributions are especially welcome around:

* **Runtime backends**: Additional backends (Ollama, MLC-LLM, TensorRT-LLM)
* **Deployment targets**: Platform support (Jetson, Apple MLX, Android QNN, WebAssembly)
* **Fleet management**: A/B deployment tools and control planes
* **Benchmarking**: Validation frameworks and performance measurement
* **Model support**: Testing with new GGUF-compatible architectures
* **Documentation**: Deployment guides, tutorials, and best practices
* **Security**: Compliance features, signed manifests, SBOM generation

## Contributing Runtime Backends

Alki's core value is providing flexible deployment options through runtime backends. When contributing new backends:

1. **Implement the BaseRuntime interface** (see `src/backends/base_runtime.py`)
2. **Add comprehensive tests** including bundle loading and inference validation
3. **Include benchmarks** comparing to existing backends (speed, memory, compatibility)
4. **Document optimal use cases** and hardware requirements for your backend
5. **Update the CLI** to support your backend selection
6. **Provide deployment examples** in documentation

See `src/backends/` for examples of proper backend implementation.

### Runtime Backend Guidelines

- Focus on **production readiness** (reliability, performance, monitoring)
- Target **edge deployment** scenarios (resource constraints, offline capability)
- Prioritize **standards compliance** (OpenAI API compatibility where possible)
- Include **hardware acceleration** support (GPU, NPU, specialized chips)
- Measure **real-world performance** on target edge devices

## 🔌 Backend Plugins

Each backend (e.g. `llama_cpp`, `ollama`, `mlc_llm`, `onnx_runtime`) is a self-contained module that exposes:

```python
def serve(bundle_path, config) -> Runtime:
    """Load a GGUF bundle and return a Runtime instance for inference."""

def containerize(bundle_path, config) -> DockerImage:
    """Generate container image for the bundle with this runtime."""
```

This makes it easy to add new runtime targets without touching the core framework.

## 🧪 Validation

All contributions should include:

* **Unit tests (pytest)** for core logic
* **Integration test** for at least one model (e.g., GPT-2 with W8A8 quant on CPU)
* **Benchmark logs** (optional, but encouraged for backend contributions)

### GGUF Conversion Testing

The project includes comprehensive GGUF conversion and quantization testing:

* **Unit tests**: `pytest tests/test_converter.py` (fast, <5 seconds)
* **End-to-end test**: `python scripts/test_pack_e2e.py` (slower, ~2 minutes)
* **Quick demo**: `python scripts/demo_gguf_pipeline.py` (shows conversion process)

When contributing conversion-related changes:
- Ensure unit tests pass for HF→GGUF conversion validation
- Run the end-to-end test to verify compatibility with real models
- Test with different quantization profiles (Q4_K_M, Q5_K_M, Q8_0) to verify output quality

### Runtime Testing

The project includes comprehensive runtime inference testing:

* **Unit tests**: `pytest tests/test_runtime.py` (fast, <5 seconds)
* **Integration test**: Test with actual model bundles using the `alki run` command
* **Manual testing**: Use different generation parameters to validate functionality

When contributing runtime-related changes:
- Ensure unit tests pass for configuration validation and text generation logic
- Test with both quantized and non-quantized model bundles
- Validate generation with different sampling parameters (temperature, top-p, top-k)
- Test error handling for missing models and invalid configurations

**Example runtime testing workflow**:
```bash
# Create a test bundle first
alki pack --hf "Qwen/Qwen3-0.6B-Instruct" --quant "Q4_K_M" --out ./dist/qwen3-0.6b

# Validate the bundle
alki validate --bundle ./dist/qwen3-0.6b

# Test basic inference with llama-server
llama-server -m ./dist/qwen3-0.6b/models/qwen3-0_6b-instruct-q4_k_m.gguf \
  --api --host 0.0.0.0 --port 8080 --ctx-size 4096

# Test with curl
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3-0.6b-instruct", "messages": [{"role": "user", "content": "Hello!"}]}'

# Test container build
alki image build --bundle ./dist/qwen3-0.6b --runtime llama.cpp --tag test/qwen3:Q4
docker run -p 8080:8080 test/qwen3:Q4
```

## 📜 License

By contributing, you agree that your contributions will be licensed under the [Apache-2.0 License](LICENSE).

## 💬 Communication

* Use GitHub Issues for bugs/feature requests
* Pull Requests for contributions
* (Future) Discussions forum/Discord for community chatter