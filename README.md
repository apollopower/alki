# Alki 🌊

**A toolchain for deploying and managing LLMs at the edge.**

Alki takes a Hugging Face model, converts it to GGUF format, applies quantization, and produces production-ready deployment bundles that run efficiently on edge devices. Supports single-device deployments and fleet-scale orchestration.

## ✨ Goals

* **Simple**: One command from HuggingFace to optimized GGUF bundle.
* **Portable**: CPU/GPU support via llama.cpp runtime with broad hardware compatibility.
* **Production-ready**: Containers, systemd units, and deployment manifests included.
* **Fleet Intelligence**: Orchestrate deployments across hundreds of edge devices with A/B testing.
* **Edge-Native**: Real-time monitoring and optimization without cloud dependency.

## 🚀 Quickstart

```bash
# Validate GGUF models with benchmarking
alki validate "Qwen/Qwen3-0.6B-GGUF" --filename "*Q8_0.gguf" --benchmark

# Convert HF model to GGUF and create deployment bundle
alki pack "Qwen/Qwen2-0.5B" --quantize Q8_0 --name qwen2-model

# Build and test container image
alki image build ./dist/qwen2-model --tag mymodel:latest
alki image test mymodel:latest

# Publish for fleet deployment
alki publish ./dist/qwen2-model --registry myregistry.com/ai --tag v1.0
```

## 📍 Status & Roadmap

**✅ Available Now:**
- Direct HF → GGUF conversion (Q8_0 quantization)
- Pre-converted GGUF model support (all quantization profiles)
- Performance benchmarking (tokens/sec, memory usage)
- Production bundles with manifests, SBOMs, and deployment configs
- Container images with llama-server runtime
- Multi-platform deployment (Docker, K8s, systemd)
- CLI: `validate`, `pack`, `image`, `publish` commands

**🚧 Phase 1 (In Progress):**
- Advanced quantization (Q4_K_M, Q5_K_M)
- Hardware optimization profiles
- End-to-end validation pipeline

**🚀 Phase 2 (Planned):**
- Multi-runtime backends (Ollama, MLC-LLM, ONNX)
- Multi-modal model support
- Hardware-specific optimization

**See [ROADMAP.md](ROADMAP.md) for complete development plan.**

## 🚀 Deployment

**Embedded Images** (simple): `alki image build` creates containers with models baked in
**Bundle Registry** (fleet): `alki publish` for efficient updates and A/B testing across devices

## 🛠️ Installation

```bash
python -m venv .venv
source .venv/bin/activate
make install
```

## 🎮 Runtime Inference

Once deployed, your models serve an OpenAI-compatible API via llama-server:

```bash
# Test with curl
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-0.6b-instruct",
    "messages": [{"role": "user", "content": "Tell me about Alki beach in Seattle, WA"}],
    "max_tokens": 100,
    "temperature": 0.8
  }'

# Or use any OpenAI client library
import openai
client = openai.OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="qwen3-0.6b-instruct",
    messages=[{"role": "user", "content": "Tell me about Alki beach in Seattle, WA"}],
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

Alki supports models through two paths:

**Pre-converted GGUF** (all quantizations): Qwen, Llama, Mistral, Phi families
**Direct HF → GGUF** (Q8_0): Qwen, Llama, Mistral, Phi, Gemma, TinyLlama, StableLM

### Quantization Profiles

| Profile | Size | Quality | Use Case |
|---------|------|---------|----------|
| Q4_K_M  | ~75% smaller | Good | Edge deployment |
| Q5_K_M  | ~65% smaller | Better | Balanced |
| Q8_0    | ~50% smaller | Excellent | Development/testing |

*Architecture support depends on llama.cpp version (b4481). Install conversion: `pip install alki[convert]`*

## 📦 Bundle Structure

Bundles include models, metadata, and deployment configs:

```
dist/my-model/
  models/*.gguf              # Quantized GGUF models
  metadata/                  # Manifest, SBOM, checksums
  deploy/
    systemd/*.service        # systemd units
    k3s/*.yaml              # Kubernetes manifests
    docker/Dockerfile       # Container config
```

**Deploy**: `kubectl apply -f ./dist/my-model/deploy/k3s/` or `cp deploy/systemd/*.service /etc/systemd/system/`



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

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
make install

# Run checks
make all     # Format, lint, and test
make check   # CI checks only
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 📜 License

### Apache-2.0

Free to use, modify, and contribute.

## 🤝 Contributing

Alki is actively developed and contributions are very welcome!

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.