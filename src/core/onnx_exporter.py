from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional

import onnx
from optimum.onnxruntime import ORTModelForCausalLM

from .constants import (
    Defaults,
    ExecutionProviders,
    SupportedArchitectures,
    get_default_dynamic_axes,
)


@dataclass
class OnnxExportConfig:
    """Configuration for ONNX export.

    Attributes:
        opset_version: ONNX opset version for compatibility
        use_gpu: Whether to use GPU provider for export
        optimize: Whether to apply ONNX optimizations during export
        output_dir: Optional custom output directory
        use_cache: Whether to enable KV cache in exported model.
                   False is recommended for quantization and edge deployment
                   as it simplifies the model and reduces memory overhead.
        dynamic_axes: Axes that can have variable sizes at runtime
    """

    opset_version: int = Defaults.ONNX_OPSET_VERSION
    use_gpu: bool = False
    optimize: bool = True
    output_dir: Optional[Path] = None
    use_cache: bool = False  # Disabled by default for edge deployment

    # Dynamic axes for variable sequence lengths
    dynamic_axes: Dict[str, Dict[int, str]] = None

    def __post_init__(self):
        if self.dynamic_axes is None:
            self.dynamic_axes = get_default_dynamic_axes(self.use_cache)


class OnnxExporter:
    """Exports HuggingFace models to ONNX format."""

    def __init__(self, config: Optional[OnnxExportConfig] = None):
        self.config = config or OnnxExportConfig()

    def export(
        self, model_artifacts: Dict[str, Any], output_path: Path
    ) -> Dict[str, Any]:
        """
        Export a HuggingFace model to ONNX format.

        Args:
            model_artifacts: Output from HuggingFaceModelLoader.prepare()
            output_path: Directory to save the ONNX model

        Returns:
            Dict with ONNX model path and export metadata
        """
        model_id = model_artifacts["model_id"]
        architecture = model_artifacts["architecture"]

        print(f"Exporting {model_id} ({architecture}) to ONNX...")

        # Ensure output directory exists
        output_path.mkdir(parents=True, exist_ok=True)

        try:
            # Use optimum for ONNX export - handles most transformer architectures
            ort_model = ORTModelForCausalLM.from_pretrained(
                model_id,
                export=True,
                use_cache=self.config.use_cache,
                provider=(
                    ExecutionProviders.CPU
                    if not self.config.use_gpu
                    else ExecutionProviders.CUDA
                ),
            )

            # Save the ONNX model
            onnx_path = output_path / "model.onnx"
            ort_model.save_pretrained(output_path)

            # Validate the exported ONNX model
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)

            print(f"✓ ONNX export successful: {onnx_path}")

            return {
                "model_id": model_id,
                "architecture": architecture,
                "onnx_path": onnx_path,
                "onnx_model": onnx_model,
                "output_dir": output_path,
                "export_config": self.config,
            }

        except Exception as e:
            error_msg = f"ONNX export failed for {model_id}: {str(e)}"
            print(f"✗ {error_msg}")

            if any(
                unsupported in architecture.lower()
                for unsupported in SupportedArchitectures.UNSUPPORTED
            ):
                error_msg += f"\nNote: {architecture} may not be fully supported by optimum ONNX export."

            raise RuntimeError(error_msg) from e

    def validate_architecture(self, architecture: str) -> bool:
        """
        Check if an architecture is known to be supported.

        Args:
            architecture: Model architecture name

        Returns:
            True if likely supported, False if known unsupported
        """
        arch_lower = architecture.lower()

        if any(arch in arch_lower for arch in SupportedArchitectures.UNSUPPORTED):
            return False

        return any(arch in arch_lower for arch in SupportedArchitectures.SUPPORTED)
