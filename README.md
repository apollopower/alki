# Alki 🌊

**A toolchain for deploying and managing LLMs at the edge with llama.cpp.**

Alki takes a Hugging Face model, converts it to GGUF format, applies quantization, and produces production-ready deployment bundles that run efficiently on edge devices. Supports single-device deployments and fleet-scale orchestration.

## ✨ Goals

* **Simple**: One command from HuggingFace to optimized GGUF bundle.
* **Portable**: CPU/GPU support via llama.cpp runtime with broad hardware compatibility.
* **Production-ready**: Containers, systemd units, and deployment manifests included.
* **Fleet Intelligence**: Orchestrate deployments across hundreds of edge devices with A/B testing.
* **Edge-Native**: Real-time monitoring and optimization without cloud dependency.

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

# Create deployment bundles from pre-converted GGUF
alki pack "Qwen/Qwen3-0.6B-GGUF" --filename "*Q8_0.gguf" --name my-model

# Or convert directly from HuggingFace PyTorch models
alki pack "Qwen/Qwen2-0.5B" --quantize Q8_0 --name qwen2-model

# Build container images
alki image build ./dist/my-model --tag mymodel:latest
alki image test mymodel:latest

# Publish bundles for fleet deployment
alki publish ./dist/my-model --local                          # Local build only
alki publish ./dist/my-model --registry myregistry.com/ai    # Push to registry

# With custom options
alki validate "Qwen/Qwen2-0.5B-Instruct-GGUF" -f "*q8_0.gguf" \
  --prompt "Explain machine learning" --max-tokens 50 --no-cleanup
```

## 📍 Current Development Status

Alki supports both **pre-converted GGUF models** from HuggingFace (e.g., `Qwen/Qwen3-0.6B-GGUF`) and **direct HuggingFace model conversion** for supported architectures.

**✅ Available now:**
- **Pre-converted GGUF Models** - Full support with all quantization profiles (Q4_K_M, Q5_K_M, Q8_0)
- **Direct HF → GGUF Conversion** - Convert supported PyTorch models (Qwen2, Llama, Mistral, etc.) with Q8_0 quantization
- **GGUF Model Validation** - Comprehensive testing with inference validation
- **Production Bundles** - Complete deployment packages with manifests and SBOMs
- **Multi-Platform Deployment** - Docker, Kubernetes, and systemd configurations
- **Container Images** - Optimized images with llama-server runtime
- **Registry Publishing** - Fleet deployment with version management
- **CLI Interface** - `validate`, `pack`, `image`, and `publish` commands

**🚧 Coming in Phase 1:**
- **Advanced Quantization Support** - Q4_K_M/Q5_K_M for optimal edge performance
- **Emulated Performance Benchmarking** - Estimate performance before deployment
- **Hardware Optimization Profiles** - Automatic tuning for common edge devices

## 🎯 Development Roadmap

**See [ROADMAP.md](ROADMAP.md) for complete development plan and contribution opportunities.**

### Phase 1: Core Functionality ⚡ *In Development*

**Model Conversion & Optimization**
- [x] Pre-converted GGUF model support
- [x] **Direct HF → GGUF conversion** (Q8_0 quantization, supported architectures)
- [x] **Pluggable converter architecture**
- [ ] **Extended architecture support** (newer models as llama.cpp evolves)
- [ ] **Architecture detection & optimization**

**Production Deployment**
- [x] Bundle format with manifests and deployment configs
- [x] Container image generation and registry publishing
- [x] Multi-platform support (Docker, K8s, systemd)
- [ ] **Performance benchmarking framework** - Evaluate models before edge deployment
- [ ] **End-to-end validation pipeline**

### Phase 2: Advanced Capabilities 🚀 *Planned*

**Enhanced Model Support**
- [ ] Multi-runtime backends (Ollama, MLC-LLM, ONNX Runtime)
- [ ] Additional model architectures as llama.cpp evolves
- [ ] Custom quantization profiles and calibration datasets
- [ ] Multi-modal model support (vision-language models)

**Developer Experience**
- [ ] Hardware-specific optimization (Jetson, Apple Silicon, x86)
- [ ] Advanced deployment configurations and templates
- [ ] Performance profiling and optimization tools
- [ ] Integration with popular development workflows

**See [ROADMAP.md](ROADMAP.md) for complete roadmap and Phase 3 ecosystem plans.**

## 🏗️ Deployment Architecture

Alki supports two deployment approaches for different use cases:

### Single-Stage: Embedded Models (`alki image`)
```bash
alki image build ./dist/my-model --tag mymodel:latest
docker run -p 8080:8080 mymodel:latest
```
- **Use case**: Simple deployments, development, single-model scenarios
- **Trade-off**: Larger images, full rebuild for model updates

### Two-Stage: Bundle Registry (`alki publish`)
```bash
# Publish bundle to registry
alki publish ./dist/my-model --registry myregistry.com/bundles --tag v1.0

# Deploy runtime container that pulls bundle
docker run -p 8080:8080 \
  -e ALKI_BUNDLE_URI=myregistry.com/bundles/my-model:v1.0 \
  myregistry.com/alki-runtime:latest
```
- **Use case**: Production fleets, A/B testing, frequent model updates
- **Benefits**: Efficient updates, version management, gradual rollouts

### Bundle Registry vs Container Registry
- **Container Registry**: Stores complete container images with models baked in
- **Bundle Registry**: Stores only model bundles; runtime containers pull them at startup
- **Result**: Faster updates, better bandwidth utilization, easier A/B deployment

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

# Publish bundle for fleet deployment
alki publish ./dist/qwen3-0.6b --registry myregistry.com/ai --tag v1.0.0
# Or build locally for testing
alki publish ./dist/qwen3-0.6b --local

# Run locally with llama-server
llama-server \
  -m ./dist/qwen3-0.6b/models/qwen3-0_6b-instruct-q4_k_m.gguf \
  --api --host 0.0.0.0 --port 8080 --ctx-size 4096

# Or run with Docker
docker run -p 8080:8080 acme/qwen3-0.6b:Q4

# Use deployment configs from bundle
cp ./dist/qwen3-0.6b/deploy/systemd/*.service /etc/systemd/system/
```

## 🎮 Runtime Inference

Once deployed, your models serve an OpenAI-compatible API via llama-server:

```bash
# Test with curl
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-0.6b-instruct",
    "messages": [{"role": "user", "content": "Tell me about Alki beach in Seattle, WA?"}],
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

### Pre-converted GGUF Models
All pre-converted GGUF models from HuggingFace work with full quantization support:

* **Qwen family** - Qwen2.5-GGUF, Qwen3-GGUF series (excellent edge performance) ✅
* **Llama family** - Llama 2-GGUF, Llama 3/3.1/3.2-GGUF (all sizes) ✅
* **Mistral family** - Mistral-7B-GGUF, Mistral-Nemo-GGUF, Codestral-GGUF ✅
* **Phi family** - Phi-3-GGUF, Phi-3.5-GGUF (Microsoft's efficient models) ✅

### Direct HF → GGUF Conversion
For direct PyTorch model conversion (Q8_0 quantization):

* **Qwen family** - Qwen2, Qwen2.5 series ✅
* **Llama family** - Llama 2, Llama 3/3.1/3.2 (all sizes) ✅
* **Mistral family** - Mistral 7B, Mistral Nemo, Codestral ✅
* **Phi family** - Phi-3, Phi-3.5 (Microsoft's efficient models) ✅
* **Gemma family** - Gemma 2B, 7B, 9B, 27B ✅
* **TinyLlama** - Popular 1.1B parameter model ✅
* **StableLM** - Stability AI's open models ✅

### Quantization Profiles

**For Pre-converted GGUF Models:**
* **Q4_K_M**: Edge-friendly default, ~4-bit quantization with good quality ✅
* **Q5_K_M**: Better quality, slightly larger size ✅
* **Q8_0**: High quality, larger size, good for development/benchmarking ✅

**For Direct HF → GGUF Conversion:**
* **Q8_0**: High quality, currently supported for direct conversion ✅
* **Q4_K_M, Q5_K_M**: Planned for Phase 2 (requires two-step quantization)

### Requirements

**For Pre-converted GGUF Models:**
* Any GGUF model from HuggingFace or local files
* Proper tokenizer configuration (usually included)

**For Direct HF → GGUF Conversion:**
* Model architecture supported by llama.cpp (current version: b4481)
* HuggingFace models with proper tokenizer configuration
* PyTorch or SafeTensors model weights

**General:**
* Respects model licensing and gating requirements
* Install conversion dependencies: `pip install alki[convert]`

**Note**: Architecture support for direct conversion depends on llama.cpp capabilities. Pre-converted GGUF models work regardless of architecture.

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

## 🚀 Deployment Configurations

Alki automatically includes production-ready deployment configurations in every bundle:

### Bundle Structure
```
dist/my-model/
  deploy/
    systemd/
      alki-my-model.service    # Systemd service file
    k3s/
      deployment.yaml          # Kubernetes manifests
    docker/
      Dockerfile              # Docker configuration
```

### Using Deployment Configs

**Systemd:**
```bash
sudo cp ./dist/my-model/deploy/systemd/*.service /etc/systemd/system/
sudo systemctl enable alki-my-model.service
sudo systemctl start alki-my-model.service
```

**Kubernetes:**
```bash
kubectl apply -f ./dist/my-model/deploy/k3s/
```

**Docker:**
```bash
docker build -f ./dist/my-model/deploy/docker/Dockerfile -t my-model .
```

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

Alki is actively developed and contributions are very welcome!

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.