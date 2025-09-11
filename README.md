# Alki 🌊

**An open-source toolchain for deploying LLMs at the edge with llama.cpp.**

Alki takes a Hugging Face model, converts it to GGUF format, applies quantization (Q4_K_M, Q5_K_M, Q8_0), and produces production-ready deployment bundles that run efficiently on edge devices with minimal dependencies.

## ✨ Goals

* **Simple**: One command from HuggingFace to optimized GGUF bundle.
* **Portable**: CPU/GPU support via llama.cpp runtime with broad hardware compatibility.
* **Production-ready**: Containers, systemd units, and deployment manifests included.
* **A/B safe**: Versioned bundles with manifests for reliable fleet deployments.

## ✅ Current Capabilities

Alki currently provides:

* **GGUF Model Validation** - Validate pre-converted GGUF models from HuggingFace or local files
* **Bundle Creation** - Package GGUF models with manifests and deployment configs
* **Container Image Generation** - Build Docker images with llama-server runtime
* **Model Loading & Inference** - Load and test GGUF models using llama-cpp-python
* **CLI Interface** - Complete toolchain with `validate`, `pack`, and `image` commands
* **Development Tools** - Test scripts for validation and model loading

```bash
# Validate GGUF models
alki validate "Qwen/Qwen3-0.6B-GGUF" --filename "*Q8_0.gguf"
alki validate /path/to/local/model.gguf

# Create deployment bundles
alki pack "Qwen/Qwen3-0.6B-GGUF" --filename "*Q8_0.gguf" --name my-model

# Build container images
alki image build ./dist/my-model --tag mymodel:latest
alki image test mymodel:latest

# With custom options
alki validate "Qwen/Qwen2-0.5B-Instruct-GGUF" -f "*q8_0.gguf" \
  --prompt "Explain machine learning" --max-tokens 50 --no-cleanup
```

## 📝 Current Limitations

Alki currently works with **pre-converted GGUF models** from HuggingFace (e.g., `Qwen/Qwen3-0.6B-GGUF`). 
Direct conversion from standard HuggingFace models is the final Phase 1 milestone.

**✅ What works today:**
- Packaging pre-converted GGUF models from HuggingFace
- Validating and testing GGUF models
- Creating production-ready bundles with manifests and SBOMs
- Generating deployment configs (Docker, K8s, systemd)
- Building container images with llama-server runtime
- Bundle verification and integrity checks

**🔜 Coming in Phase 1 completion:**
- Direct HF model → GGUF conversion with pluggable architecture
- Actual quantization conversion (currently preserves original quantization)
- Multiple quantization profiles in one command (Q4_K_M,Q5_K_M,Q8_0)
- Automatic architecture detection and optimization
- Deployment recipe generation (`alki recipe emit`)

## 🗺️ Roadmap (Phase 1)

**Model Ingestion:**
- [x] Pre-converted GGUF model support
- [ ] Direct HF → GGUF conversion
- [ ] Pluggable converter architecture

**Bundle Management:**
- [x] Bundle format (manifests + GGUF models + deployment configs)
- [x] Bundle verification and integrity checks
- [x] SBOM generation

**CLI Commands:**
- [x] `alki validate` - GGUF model validation
- [x] `alki pack` - Bundle creation from GGUF models
- [x] `alki image` - Container image generation
- [ ] `alki publish` - Bundle registry publishing  
- [ ] `alki recipe` - Deployment recipe generation

**Runtime Integration:**
- [x] GGUF model loading with llama-cpp-python
- [x] Model capability extraction
- [ ] llama-server integration and wrapping

**Deployment Support:**
- [x] Systemd service configs
- [x] Docker container configs
- [x] Kubernetes manifests
- [ ] Nomad job specs

**Validation & Testing:**
- [x] GGUF model validation
- [x] Bundle integrity verification
- [ ] Performance benchmarking
- [ ] End-to-end smoke tests

## 🗺️ Roadmap (Phase 2)

* [ ] Additional runtime backends as plugins
  * [ ] Ollama integration (auto-generate Modelfile)
  * [ ] MLC-LLM support (TVM packages)
  * [ ] ONNX Runtime option (backward compatibility)
  * [ ] Manual quantization methods (Q4_K_M, Q5_K_M, Q8_0)
* [ ] Fleet management and A/B deployment tools

## 🚀 Quickstart

```bash
# Setup
python -m venv .venv
source .venv/bin/activate
make install

# Pack a pre-converted GGUF model for deployment
alki pack "Qwen/Qwen3-0.6B-GGUF" \
  --filename "*Q8_0.gguf" \
  --context-size 4096 \
  --out ./dist/qwen3-0.6b

# Validate a GGUF model from HuggingFace with custom context size
alki validate "Qwen/Qwen3-0.6B-GGUF" --filename "*Q8_0.gguf" --context-size 1024

# Build container image from bundle
alki image build ./dist/qwen3-0.6b --tag acme/qwen3-0.6b:latest

# Test the container image
alki image test acme/qwen3-0.6b:latest

# List available images
alki image list

# Run locally with llama-server
llama-server \
  -m ./dist/qwen3-0.6b/models/qwen3-0_6b-instruct-q4_k_m.gguf \
  --api --host 0.0.0.0 --port 8080 --ctx-size 4096

# Or run with Docker
docker run -p 8080:8080 acme/qwen3-0.6b:Q4

# Generate deployment recipes
alki recipe emit \
  --bundle ./dist/qwen3-0.6b \
  --target systemd \
  --out ./dist/deploy/systemd
```

## 🎮 Runtime Inference

Once deployed, your models serve an OpenAI-compatible API via llama-server:

```bash
# Test with curl
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-0.6b-instruct",
    "messages": [{"role": "user", "content": "Hello, how are you?"}],
    "max_tokens": 100,
    "temperature": 0.8
  }'

# Or use any OpenAI client library
import openai
client = openai.OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="qwen3-0.6b-instruct",
    messages=[{"role": "user", "content": "Hello, how are you?"}],
    max_tokens=100
)
```

### Health Checks

```bash
# Check model status
curl http://localhost:8080/v1/models

# Health endpoint
curl http://localhost:8080/health
```

## 🤖 Supported Models

### GGUF Compatible Models
Most modern transformer models with good llama.cpp support work out-of-the-box:

* **Qwen family** - Qwen2.5, Qwen3 series (excellent edge performance) ✅
* **Llama family** - Llama 2, Llama 3/3.1/3.2 (all sizes) ✅
* **Mistral family** - Mistral 7B, Mistral Nemo, Codestral ✅
* **Phi family** - Phi-3, Phi-3.5 (Microsoft's efficient models) ✅
* **Gemma family** - Gemma 2B, 7B, 9B, 27B ✅
* **TinyLlama** - Popular 1.1B parameter model ✅
* **StableLM** - Stability AI's open models ✅

### Quantization Profiles

* **Q4_K_M**: Edge-friendly default, ~4-bit quantization with good quality
* **Q5_K_M**: Better quality, slightly larger size
* **Q8_0**: High quality, larger size, good for development/benchmarking

### Requirements

* Models must have llama.cpp conversion support
* HuggingFace models with proper tokenizer configuration
* Respects model licensing and gating requirements

**Note**: Most modern transformer architectures work well. MoE (Mixture of Experts) and state-space models may have limited support depending on llama.cpp capabilities.

## 📦 Bundle Layout

```
dist/qwen3-0.6b/
  models/
    qwen3-0_6b-instruct-q4_k_m.gguf    # Q4_K_M quantized model (~350MB)
    qwen3-0_6b-instruct-q5_k_m.gguf    # Q5_K_M quantized model (~420MB)
    qwen3-0_6b-instruct-q8_0.gguf      # Q8_0 quantized model (~640MB)
  metadata/
    manifest.json                       # Bundle metadata, hashes, capabilities
    sbom.spdx.json                     # Software bill of materials
    LICENSE.txt                        # Model license
    README.md                          # Quick start guide
  deploy/
    systemd/
      alki-qwen3.service              # systemd unit file
    k3s/
      deployment.yaml                 # Kubernetes deployment
      service.yaml                    # Kubernetes service
    docker/
      Dockerfile                      # Container image definition
```

## 🚀 Deployment Recipes

Alki generates production-ready deployment configurations for various platforms:

### Systemd (Bare Metal/Linux)
```bash
alki recipe emit --bundle ./dist/qwen3-0.6b --target systemd --out ./deploy
```
Generates: `systemd/alki-qwen3.service` with proper service configuration, auto-restart, and resource limits.

### Kubernetes/k3s
```bash
alki recipe emit --bundle ./dist/qwen3-0.6b --target k3s --out ./deploy
```
Generates: Deployment, Service, ConfigMap with health checks and horizontal pod autoscaling.

### Docker/Containers
```bash
alki image build --bundle ./dist/qwen3-0.6b --runtime llama.cpp --tag acme/qwen3:Q4
```
Creates optimized container images with llama-server, health endpoints, and proper security context.

## 📋 Manifests

### Model Manifest (manifest.json)
```json
{
  "name": "qwen3-0.6b-instruct",
  "version": "2025-09-06.1",
  "artifacts": [
    {"quant":"Q4_K_M","uri":"./models/qwen3-0_6b-instruct-q4_k_m.gguf","sha256":"abc123...","size":367001600},
    {"quant":"Q5_K_M","uri":"./models/qwen3-0_6b-instruct-q5_k_m.gguf","sha256":"def456...","size":441450496}
  ],
  "defaults": {"ctx":4096,"threads":"auto","ngl":0},
  "template":"qwen3",
  "license":"apache-2.0"
}
```

### Runtime Manifest
```json
{
  "runtime": "llama.cpp",
  "server": {"host":"0.0.0.0","port":8080,"api":true},
  "args": {"ctx":4096,"threads":"auto","ngl":0},
  "health": {"path":"/v1/models","timeout_s":5}
}
```

## 🎯 GGUF Quantization Profiles

Alki uses llama.cpp's battle-tested quantization methods:

### Quantization Options

- **Q4_K_M**: 4-bit quantization, edge-friendly default with excellent quality/size balance
- **Q5_K_M**: 5-bit quantization, better quality with moderate size increase
- **Q8_0**: 8-bit quantization, high quality for development and benchmarking

### Performance Characteristics

| Profile | Size Reduction | Inference Speed | Quality Loss | Use Case |
|---------|----------------|-----------------|--------------|----------|
| Q4_K_M  | ~75%          | Fastest         | Minimal      | Edge deployment |
| Q5_K_M  | ~65%          | Fast            | Very low     | Balanced performance |
| Q8_0    | ~50%          | Good            | Nearly none  | Development/testing |

### Auto-Profiling (Future)

Alki will include a `calibrate` command that:
* Detects RAM/CPU/AVX capabilities and available GPU+VRAM
* Runs benchmarks on different quantization profiles
* Recommends optimal settings for your hardware
* Generates per-host configuration profiles

**Example**: `alki calibrate --model qwen3-0.6b --save-profile edge-device.json`

## 🛠️ Tech Stack

* **llama.cpp** - Core runtime with broad CPU/GPU compatibility
* **Python 3.10+** - CLI, conversion pipeline, and bundle generation
* **Typer** - Command-line interface
* **GGUF** - Model format with efficient quantization
* **Docker** - Container packaging and distribution
* **Pytest** - Testing and validation harness

### Optional Performance Components (Future)
* **Go/Rust** - Performance-critical packaging operations
* **Sigstore/cosign** - Signed manifests and artifact verification

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

# Test GGUF model validation
python scripts/test_validator.py

# Test specific GGUF models
python scripts/test_validator.py --repo-id "Qwen/Qwen3-0.6B-GGUF" --filename "*Q8_0.gguf"
python scripts/test_validator.py --repo-id "Qwen/Qwen2-0.5B-Instruct-GGUF" --filename "*q8_0.gguf" --no-cleanup

# Test with custom context size
python scripts/test_validator.py --repo-id "Qwen/Qwen3-0.6B-GGUF" --filename "*Q8_0.gguf" --context-size 2048

# Test model loading and inference
python scripts/test_llama.py "Qwen/Qwen2-0.5B-Instruct-GGUF" "*q8_0.gguf"
python scripts/test_llama.py "Qwen/Qwen3-0.6B-GGUF" "*Q8_0.gguf" --no-cleanup

# Test end-to-end packing pipeline with real models (future)
python scripts/test_pack_e2e.py

# Test with different models and quantization profiles (Phase 1 completion - HF conversion)
python scripts/test_pack_e2e.py --hf Qwen/Qwen3-0.6B-Instruct --quant Q4_K_M,Q5_K_M

# Test container image generation (future)
python scripts/test_image_build.py

# Quick demo of GGUF conversion and quantization (future)
python scripts/demo_gguf_pipeline.py
```

**Important**: All commands require an activated virtual environment:
```bash
source .venv/bin/activate  # Required before any make or python commands
```

### Security & Compliance

* **License compliance**: Respects HuggingFace model licensing and gating
* **Signed manifests**: Support for Sigstore/cosign verification (future)
* **SBOM generation**: Software bill of materials for all bundles
* **Checksum verification**: SHA256 hashes for all artifacts

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed contribution guidelines.

## 📜 License

### Apache-2.0

Free to use, modify, and contribute.

## 🤝 Contributing

Alki is just getting started. Contributions are welcome, especially around:

* Additional runtime backends (Ollama, MLC-LLM, TensorRT-LLM)
* Deployment targets (Jetson, Apple MLX, Android QNN, WebAssembly)
* Fleet management and A/B deployment tools
* Benchmarking and validation frameworks
* Security and compliance features