# Alki Roadmap - Simplified & Focused

## Mission
The simplest way to deploy HuggingFace models to edge devices with world-class quantization.

## Core Vision
"The easiest way to deploy HuggingFace models to edge devices with best-in-class quantization"

One command to go from HuggingFace → Optimized edge bundle with benchmarks.

## Phase 1: Quantization Excellence + Quality of Life

### Multiple Quantization Methods
- [ ] AWQ (Activation-aware Weight Quantization) implementation
- [ ] GPTQ implementation with configurable bit width (4/8 bit)
- [ ] Unified quantizer factory for method selection
- [ ] Quantization comparison tool

### Built-in Benchmarking Suite
- [ ] Automatic benchmarking during build process
- [ ] Perplexity evaluation against baseline
- [ ] Speed profiling (tokens/second)
- [ ] Memory usage tracking
- [ ] Size reduction metrics
- [ ] Comparison report generation

### Enhanced CLI Experience
```bash
# Example workflows
alki build tinyllama --quantize awq --benchmark
alki compare tinyllama --methods all
alki info dist/tinyllama-awq  # Rich bundle information
```

### Quality of Life Improvements
- [ ] Rich bundle information display with performance metrics
- [ ] Automatic perplexity checking to catch bad quantizations
- [ ] Better error messages with actionable suggestions
- [ ] Progress bars for long operations
- [ ] Comparison tables for quantization methods

## Phase 2: Hardware Optimization (Future)

### Device Presets
- [ ] Raspberry Pi optimization profiles
- [ ] NVIDIA Jetson optimization profiles
- [ ] Intel NUC optimization profiles
- [ ] Auto-detection of hardware capabilities

### Auto-optimization
- [ ] Detect hardware and select optimal quantization
- [ ] Hardware-specific kernel selection
- [ ] Memory-aware configuration

## Success Metrics
- ✅ 3+ quantization methods (SmoothQuant, AWQ, GPTQ)
- ✅ 5+ validated models (TinyLlama, Phi-2, Gemma-2B, Qwen-0.5B, StableLM-3B)
- ✅ Integrated benchmarking showing clear performance/quality tradeoffs
- ✅ <5 minute workflow from HuggingFace to deployed bundle
- ✅ Comparison tools help users make informed decisions

## Non-Goals (Staying Focused)
- ❌ Model training/fine-tuning - Stay inference-only
- ❌ Multi-model systems - Single model bundles only
- ❌ Cloud deployment - Pure edge focus
- ❌ Custom architectures - HuggingFace models only
- ❌ Pruning techniques - Focus on quantization first
- ❌ Distillation - No training infrastructure

## Example User Journey
```bash
# 1. Compare quantization methods
$ alki compare tinyllama --quick
> Recommendation: Use AWQ for best balance

# 2. Build with recommended settings  
$ alki build tinyllama --quantize awq --benchmark
> ✓ Built successfully
> 📊 75% smaller, 2.3x faster, 1.8% accuracy loss

# 3. Test inference
$ alki run dist/tinyllama-awq --prompt "Explain quantum computing"
> [Fast, quality response]

# 4. Deploy with confidence
$ scp -r dist/tinyllama-awq pi@raspberrypi:~/models/
```

## Technical Implementation Notes

### New Module Structure
```
src/core/
├── quantizers/
│   ├── __init__.py
│   ├── base_quantizer.py      # Abstract base class
│   ├── smoothquant.py         # Existing (refactored)
│   ├── awq_quantizer.py       # New
│   ├── gptq_quantizer.py      # New
│   └── quantizer_factory.py   # Method selection
├── evaluation/
│   ├── __init__.py
│   ├── perplexity.py          # Model quality metrics
│   ├── benchmarker.py         # Performance metrics
│   └── comparator.py          # Method comparison
```

### CLI Command Evolution
```bash
# Current
alki build model --quantize --alpha 0.5

# New
alki build model --quantize [smoothquant|awq|gptq|none] --bits [4|8] --benchmark
alki compare model --methods [all|smoothquant,awq,gptq]
alki bench dist/bundle --metrics [perplexity|speed|memory|all]
```

---

*This focused roadmap positions Alki as THE quantization toolkit for edge AI deployment - simple, powerful, and laser-focused on making small models work great on edge devices.*