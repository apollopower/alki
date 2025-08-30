# Alki 🌊

**An open-source toolchain for deploying LLMs at the edge.**

Alki takes a Hugging Face model, optimizes it, and produces a self-contained deployment bundle that can run efficiently on edge devices.

---

## ✨ Goals

* **Simple**: one command to turn a Hugging Face model into an edge-ready bundle.
* **Flexible**: quantization & optimization built in, with hardware-specific presets.
* **Portable**: bundles run with minimal dependencies on the target device.
* **Extensible**: plugin architecture for new backends (TensorRT-LLM, MLX, QNN, etc.).

---

## 🗺️ Roadmap (Phase 1)

* [ ] Model ingestion (HF → ONNX export)
* [ ] SmoothQuant W8A8 quantization pass
* [ ] Bundle format (`bundle.yaml` + tokenizer + model artifacts)
* [ ] CLI (`alki build`, `alki run`, `alki bench`)
* [ ] ORT GenAI runtime integration (CPU EP)
* [ ] OpenVINO preset (INT8 acceleration on Intel CPUs/NPUs)
* [ ] Basic validation harness (perplexity, latency, memory)

---

## 🚀 Quickstart (planned)

```bash
# Install (coming soon)
pip install alki

# Build a CPU bundle from a Hugging Face model
alki build \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --target cpu \
  --preset balanced \
  --out dist/llama3b-cpu

# Run it locally
alki run --bundle dist/llama3b-cpu --prompt "Hello from the edge!"

# Benchmark performance
alki bench --bundle dist/llama3b-cpu
```

---

## 📦 Bundle Layout (early draft)

```
dist/llama3b-cpu/
  bundle.yaml          # config + presets
  model.onnx           # quantized ONNX model
  tokenizer/           # tokenizer.json, merges.txt, vocab
  runners/             # lightweight launchers
```

---

## 🔌 Presets

* **cpu** → ONNX Runtime GenAI, CPU execution provider.
* **openvino** → ONNX Runtime GenAI + Intel OpenVINO EP with INT8 calibration.

More presets coming (TensorRT-LLM, MLX, ExecuTorch).

---

## 🛠️ Tech Stack

* **Python 3.10+** (pipelines, CLI, quantization scripts)
* **Typer** for CLI
* **ONNX / ONNX Runtime GenAI** as the first runtime
* **OpenVINO Toolkit** for Intel acceleration
* **Pytest** for validation harness
* **Docker (optional)** for reproducible builds (OpenVINO builder image planned)

---

## 📜 License

### Apache-2.0

Free to use, modify, and contribute.

---

## 🤝 Contributing

Alki is just getting started. Contributions are welcome, especially around:

* New backends (Jetson TensorRT, Apple MLX, Android QNN)
* Quantization recipes (AWQ, GPTQ, KV-cache quant)
* Benchmarking + validation tools