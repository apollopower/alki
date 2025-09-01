# Alki Backends

This directory contains backend plugins for different hardware acceleration targets.

## Architecture

Each backend implements a common interface:
- `prepare(model_ir, config) -> Bundle`
- Handles model optimization for specific hardware
- Returns standardized bundle format

## Current Backends

- [ ] **onnx_cpu** - Basic ONNX Runtime CPU execution
- [ ] **openvino** - Intel OpenVINO acceleration

## Future Backends (TODO)

### Optimum Integration
- [ ] **optimum_intel** - Leverage Optimum-Intel for OpenVINO backend
  - Use NNCF quantization
  - OpenVINO EP integration
  - INT8 calibration flows
  
- [ ] **optimum_nvidia** - Leverage Optimum-NVIDIA for TensorRT-LLM
  - TensorRT optimization
  - Multi-GPU support
  
- [ ] **optimum_habana** - Gaudi accelerator support

### Direct Integrations
- [ ] **mlx** - Apple Silicon optimization
- [ ] **executorch** - Mobile/embedded deployment
- [ ] **qnn** - Qualcomm Neural Network SDK

## Adding a New Backend

1. Create `backends/<name>_backend.py`
2. Implement `BaseBackend` interface
3. Add to backend registry
4. Update CLI to support `--target <name>`