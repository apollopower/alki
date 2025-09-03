# Alki 🌊

**An open-source toolchain for deploying LLMs at the edge.**

Alki takes a Hugging Face model, optimizes it, and produces a self-contained deployment bundle that can run efficiently on edge devices.

## ✨ Goals

* **Simple**: one command to turn a Hugging Face model into an edge-ready bundle.
* **Flexible**: quantization & optimization built in, with hardware-specific presets.
* **Portable**: bundles run with minimal dependencies on the target device.
* **Extensible**: plugin architecture for new backends (TensorRT-LLM, MLX, QNN, etc.).

## 🗺️ Roadmap (Phase 1)

* [x] Model ingestion (HF → ONNX export)
* [x] SmoothQuant W8A8 quantization pass
* [x] Bundle format (`bundle.yaml` + tokenizer + model artifacts)
* [x] CLI (`alki build`, `alki info`, `alki list`)
* [ ] Runtime commands (`alki run`, `alki bench`)
* [ ] ORT GenAI runtime integration (CPU EP)
* [ ] OpenVINO preset (INT8 acceleration on Intel CPUs/NPUs)
* [ ] Basic validation harness (perplexity, latency, memory)

## 🗺️ Roadmap (Phase 2)

* [ ] Optimum backend integration (leverage HF Optimum as building blocks)
  * [ ] Optimum-Intel for OpenVINO flows
  * [ ] Optimum-NVIDIA for TensorRT-LLM
  * [ ] Unified backend plugin interface

## 🚀 Quickstart

```bash
# Setup
python -m venv .venv
source .venv/bin/activate
make install

# Build a CPU bundle with quantization (74.6% size reduction)
python -m src.cli.main build gpt2 \
  --output dist \
  --target cpu \
  --preset balanced \
  --alpha 0.5

# Build without quantization
python -m src.cli.main build gpt2 \
  --output dist \
  --no-quantize

# View bundle information
python -m src.cli.main info dist/gpt2-cpu

# List all bundles
python -m src.cli.main list --path dist --verbose

# Runtime inference (coming soon)
# alki run --bundle dist/gpt2-cpu --prompt "Hello from the edge!"
```

## 🤖 Supported Models

### Currently Tested & Validated
* **GPT-2** (all sizes) - Reference implementation and testing

### Priority Models (Compatible, In Development)
* **Qwen3-0.6B** ⭐ - **Recommended for edge deployment** (latest generation, compact)
* **Qwen2.5-0.5B** - Proven edge performance
* **TinyLlama-1.1B** - Popular lightweight option
* **Phi-2 (2.7B)** - Microsoft's efficient edge model
* **Phi-3.5-mini** - Latest Phi generation

### Experimental Support
* **Mistral-7B** - Full-size model for testing
* **Gemma models** - Google's open models
* **StableLM variants** - Stability AI models

**Note**: All models listed use architectures supported by Optimum's ONNX export. Additional models may work but haven't been validated. See [ROADMAP.md](ROADMAP.md) for expansion plans.

## 📦 Bundle Layout

```
dist/gpt2-cpu/
  bundle.yaml            # Bundle manifest with metadata and runtime config
  model.onnx            # Quantized ONNX model (158MB from 622MB with SmoothQuant)
  model_original.onnx   # Original FP32 model (if quantized)
  tokenizer/            # tokenizer.json, tokenizer_config.json, vocab.json, merges.txt
  runners/              # Lightweight launchers (placeholder for future)
```

## 🔌 Presets

* **cpu** → ONNX Runtime GenAI, CPU execution provider.
* **openvino** → ONNX Runtime GenAI + Intel OpenVINO EP with INT8 calibration.

More presets coming (TensorRT-LLM, MLX, ExecuTorch).

## ⚡ Quantization

Alki uses **SmoothQuant W8A8** for post-training quantization, providing:

* **74.6% model size reduction** (622MB → 158MB for GPT-2)
* **2-4x faster inference** on CPUs with INT8 support
* **Minimal accuracy loss** (<1% typical)
* **No retraining required**

The `alpha` parameter controls smoothing strength:
* `α=0.0`: Baseline quantization (fastest)
* `α=0.5`: Balanced (recommended default)
* `α=1.0`: Maximum smoothing (best for outlier-heavy models)

**Note**: Warning messages during quantization are normal and expected part of the ONNX Runtime optimization process.

## 🛠️ Tech Stack

* **Python 3.10+** (pipelines, CLI, quantization scripts)
* **Typer** for CLI
* **ONNX / ONNX Runtime GenAI** as the first runtime
* **OpenVINO Toolkit** for Intel acceleration
* **Pytest** for validation harness
* **Docker (optional)** for reproducible builds (OpenVINO builder image planned)

## 🔧 Development

For contributors and developers:

```bash
# Setup development environment
python -m venv .venv
source .venv/bin/activate
make install  # or: pip install -e .[dev]

# Run all CI checks locally (recommended before pushing)
make check

# Auto-format and run all checks
make all

# Individual commands
make test          # Run tests
make format        # Format code with black
make lint          # Lint with ruff
make clean         # Clean cache files

# Test real ONNX export with actual models (separate from unit tests)
python scripts/test_onnx_export_e2e.py

# Test end-to-end quantization pipeline with real models
python scripts/test_quantization_e2e.py

# Quick demo of quantization concepts and configuration
python scripts/demo_quantization.py
```

**Important**: All commands require an activated virtual environment:
```bash
source .venv/bin/activate  # Required before any make or python commands
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed contribution guidelines.

## 📜 License

### Apache-2.0

Free to use, modify, and contribute.

## 🤝 Contributing

Alki is just getting started. Contributions are welcome, especially around:

* New backends (Jetson TensorRT, Apple MLX, Android QNN)
* Quantization recipes (AWQ, GPTQ, KV-cache quant)
* Benchmarking + validation tools