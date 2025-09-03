# Alki Development Roadmap 🗺️

This document tracks the development priorities and implementation plan for Alki.

## Current Status (September 2025)

**✅ Phase 1: Core Pipeline - MOSTLY COMPLETE**
- [x] Model ingestion (HuggingFace → Local)
- [x] ONNX export (via Optimum)
- [x] SmoothQuant W8A8 quantization 
- [x] Bundle format (yaml + artifacts)
- [x] CLI commands (`build`, `info`, `list`)
- [x] Comprehensive test suite (99 tests passing)

**❌ Missing from Phase 1:**
- [ ] Runtime commands (`run`, `bench`) 
- [ ] ORT GenAI integration for inference
- [ ] OpenVINO backend implementation
- [ ] Validation harness (accuracy, performance)

---

## 🎯 Immediate Priorities (Next Sprint)

### 1. **Expand Model Support Beyond GPT-2** 
*Status: Research Complete, Implementation Needed*

**Problem**: Currently only GPT-2 is validated, but many better edge models exist.

**Tasks**:
- [x] Update `src/core/onnx_exporter.py` architecture validation list
- [x] Add TinyLlama, Phi, StableLM, Gemma to supported architectures  
- [x] Test end-to-end pipeline with TinyLlama-1.1B (memory detection implemented)
- [ ] Create model-specific optimization presets
- [ ] Document model-specific quirks and settings

**Target Models** (Optimum-native support + memory-friendly):
- **TinyLlama-1.1B** ⭐ (Primary target - 1.1B params, native Llama support)
- microsoft/phi-2 (2.7B params, native Phi support)  
- StableLM-3B (native StableLM support)
- google/gemma-2b (native Gemma support)

**Future/Advanced Models** (require custom configs or high memory):
- Qwen3-0.6B (needs custom ONNX config + high memory)
- Qwen2.5-0.5B (needs custom config investigation)

### 2. **Runtime Implementation** (`alki run`)
*Status: Not Started*

**Problem**: No way to actually run inference with created bundles.

**Tasks**:
- [ ] Integrate ONNX Runtime GenAI for inference
- [ ] Implement basic prompt interface
- [ ] Add streaming response support
- [ ] Handle tokenizer integration properly
- [ ] Support both CPU and quantized models

**CLI Target**:
```bash
alki run --bundle dist/qwen3-cpu --prompt "Explain edge computing"
```

### 3. **Benchmark Suite** (`alki bench`)  
*Status: Not Started*

**Problem**: No way to measure model performance for edge deployment decisions.

**Tasks**:
- [ ] Memory usage tracking during inference
- [ ] Tokens/second measurement  
- [ ] Model loading time measurement
- [ ] Compare original vs quantized accuracy
- [ ] Generate comparison reports

**CLI Target**:
```bash
alki bench --bundle dist/qwen3-cpu --compare-models
```

---

## 📋 Phase 1 Completion Tasks

### 4. **OpenVINO Backend Implementation**
*Status: Stub Only*

**Current**: Empty directory with README only
**Needed**: 
- [ ] Create `src/backends/openvino_backend.py`
- [ ] Implement INT8 calibration workflow
- [ ] Add OpenVINO execution provider integration
- [ ] Create Intel CPU/NPU optimization presets

### 5. **Validation Harness** 
*Status: Not Started*

**Tasks**:
- [ ] Perplexity calculation vs original model
- [ ] Automated accuracy regression testing
- [ ] Performance benchmark standardization
- [ ] CI integration for model validation

---

## 🔮 Phase 2: Advanced Features

### Backend Plugin System
- [ ] Optimum-Intel integration (replace direct OpenVINO)
- [ ] Optimum-NVIDIA for TensorRT-LLM
- [ ] MLX backend for Apple Silicon
- [ ] ExecuTorch for mobile deployment

### Model Format Extensions  
- [ ] Local model path support (not just HF Hub)
- [ ] Custom/fine-tuned model support
- [ ] Multi-model bundles
- [ ] Model conversion utilities

### Advanced Quantization
- [ ] AWQ (Activation-aware Weight Quantization)
- [ ] GPTQ integration
- [ ] KV-cache quantization
- [ ] Dynamic quantization options

---

## 🚧 Known Technical Debt

### Code Architecture
1. **Hard-coded architecture lists** in `onnx_exporter.py` - needs dynamic detection
2. **Missing error handling** for unsupported models - should gracefully degrade
3. **Bundle validation** could be more comprehensive
4. **Test coverage** for real models (vs mocked) - need more E2E tests

### Memory Constraint Handling  
1. **Large model export limitations** - ✅ **COMPLETED: Memory management system implemented**
   - ✅ Memory usage monitoring with psutil integration
   - ✅ Automatic memory threshold detection (warning at 80%, critical at 90%)
   - ✅ Graceful failure with helpful error messages for oversized models
   - ✅ Model size estimation based on architecture parameters
   - ✅ Low memory mode configuration for environment optimization
   - ✅ Memory-managed operation context managers with cleanup
   - Issue: Qwen3-0.6B requires custom ONNX config (not natively supported by Optimum) AND significant memory
   - Current approach: Smart detection prevents OOM crashes, provides actionable feedback
   - Future enhancements:
     - External data storage configuration for large tensors
     - Chunked processing for memory-efficient exports

2. **Custom ONNX configuration framework** - For models not natively supported by Optimum
   - Need systematic approach for adding custom export configs
   - Framework for handling architecture-specific export requirements
   - Memory-aware export strategies

### Documentation  
1. **Model-specific guides** - Each model may have optimization quirks
2. **Backend development guide** - How to add new acceleration backends
3. **Performance tuning guide** - Model-specific quantization parameters
4. **Memory requirements guide** - System requirements for different model sizes

---

## 📊 Success Metrics

**Phase 1 Complete When**:
- [ ] Qwen3-0.6B fully supported (build → run → bench)
- [ ] At least 3 edge models validated and documented
- [ ] `alki run` provides acceptable inference experience
- [ ] Basic benchmarking shows quantization benefits

**Phase 2 Success**:
- [ ] Multiple backend options available
- [ ] Optimum integration reduces maintenance burden  
- [ ] Community contributions for new models/backends

---

*Last Updated: December 2024*
*Next Review: After model support expansion*