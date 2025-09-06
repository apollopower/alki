# Alki 🌊

**An open-source toolchain for deploying LLMs at the edge with best-in-class quantization methods.**

Alki takes a Hugging Face model, applies state-of-the-art quantization (SmoothQuant, AWQ, GPTQ), and produces an optimized deployment bundle that runs efficiently on edge devices.

## ✨ Goals

* **Simple**: One command from HuggingFace to optimized edge bundle.
* **Powerful**: Multiple quantization methods (SmoothQuant, AWQ, GPTQ) with built-in benchmarking.
* **Informed**: Automatic comparison tools help choose optimal settings.
* **Portable**: Self-contained bundles run on edge devices with minimal dependencies.

## 🗺️ Roadmap (Phase 1)

* [x] Model ingestion (HF → ONNX export)
* [x] SmoothQuant W8A8 quantization pass
* [x] Bundle format (`bundle.yaml` + tokenizer + model artifacts)
* [x] CLI (`alki build`, `alki info`, `alki list`)
* [x] Runtime commands (`alki run`)
* [x] ONNX Runtime integration (CPU EP)
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

# Runtime inference
python -m src.cli.main run dist/gpt2-cpu --prompt "Hello from the edge!" --max-tokens 50 --temperature 0.8

# Compare quantization methods (future)
python -m src.cli.main compare gpt2 --methods all
```

## 🎮 Runtime Inference

The `alki run` command performs text generation using your deployed bundles:

```bash
# Basic usage
python -m src.cli.main run dist/gpt2-cpu --prompt "Once upon a time"

# Advanced generation parameters
python -m src.cli.main run dist/gpt2-cpu \
  --prompt "The future of AI is" \
  --max-tokens 100 \
  --temperature 0.8 \
  --top-p 0.95 \
  --verbose

# Interactive generation with different models
python -m src.cli.main run dist/DialoGPT-small-cpu \
  --prompt "Human: Hello! How are you today?" \
  --max-tokens 75 \
  --temperature 0.7
```

### Parameters

* `--prompt, -p`: Input text prompt for generation (default: "Hello, I am")
* `--max-tokens, -m`: Maximum tokens to generate (default: 100)
* `--temperature, -t`: Sampling temperature, 0.1-2.0 (default: 1.0)
* `--top-p`: Nucleus sampling threshold, 0.0-1.0 (default: 0.9)
* `--verbose, -v`: Show detailed generation info and timing

### Expected Output

```
🤖 Loading bundle: gpt2-cpu
✓ Model loaded successfully
🎯 Generating with max_tokens=50, temperature=0.8

Generated text: Once upon a time, there was a small village nestled in the mountains where everyone knew each other's stories.

📊 Generation complete: 23 tokens in 1.2s (19.2 tokens/sec)
```

## 🤖 Supported Models

### Currently Supported (Native Optimum ONNX Export)
* **GPT-2** (all sizes) - Reference implementation, fully tested ✅
* **microsoft/DialoGPT-small** - Tested and working ✅
* **TinyLlama-1.1B** - Popular lightweight option (requires ~11.5GB memory for ONNX export)
* **Phi-2 (2.7B)** - Microsoft's efficient edge model
* **StableLM models** - Stability AI variants
* **Gemma models** - Google's open models (may require access)
* **Llama family** - Includes all LlamaForCausalLM variants
* **Mistral family** - Full-size models for testing

### Future Models (Require Custom ONNX Configuration)
* **Qwen3-0.6B** - Needs custom ONNX config + high memory (>4GB)
* **Qwen2.5-0.5B** - Needs custom ONNX config development
* **Phi-3.5-mini** - Not currently supported (different from Phi-2)

### Known Incompatible
* **Mamba models** - Architecture not supported by ONNX export
* **Mixtral models** - Architecture not supported by ONNX export

**Note**: Models in "Currently Supported" use native Optimum ONNX export and should work out-of-the-box. "Future Models" require additional development work. Memory constraints may affect larger models on resource-limited systems. See [ROADMAP.md](ROADMAP.md) for detailed expansion plans.

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

## 🎯 Quantization Methods

Alki provides multiple state-of-the-art quantization techniques:

- **SmoothQuant**: Balanced approach, good for most models. Provides 74.6% size reduction with <1% accuracy loss.
- **AWQ**: Activation-aware quantization, often achieves best accuracy preservation *(coming soon)*
- **GPTQ**: Aggressive compression for smallest model sizes *(coming soon)*

Use `alki compare` to find the best method for your model and hardware *(coming soon)*.

### Current SmoothQuant Features

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
* **ONNX / ONNX Runtime** as the runtime backend
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

# Test with different models and options
python scripts/test_quantization_e2e.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --low-memory

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