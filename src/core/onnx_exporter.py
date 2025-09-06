from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional
import logging

import onnx
from optimum.onnxruntime import ORTModelForCausalLM

from .constants import (
    Defaults,
    ExecutionProviders,
    SupportedArchitectures,
    get_default_dynamic_axes,
)
from .memory_manager import (
    MemoryManager,
    memory_managed_operation,
    estimate_model_memory_mb,
)

logger = logging.getLogger(__name__)


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
        low_memory: Whether to use memory optimization strategies
        memory_threshold_mb: Memory threshold in MB for low memory mode
        dynamic_axes: Axes that can have variable sizes at runtime
    """

    opset_version: int = Defaults.ONNX_OPSET_VERSION
    use_gpu: bool = False
    optimize: bool = True
    output_dir: Optional[Path] = None
    use_cache: bool = False  # Disabled by default for edge deployment
    low_memory: bool = False  # Enable memory optimization strategies
    memory_threshold_mb: float = 6000.0  # Threshold for enabling low memory mode

    # Dynamic axes for variable sequence lengths
    dynamic_axes: Dict[str, Dict[int, str]] = None

    def __post_init__(self):
        if self.dynamic_axes is None:
            self.dynamic_axes = get_default_dynamic_axes(self.use_cache)


class OnnxExporter:
    """Exports HuggingFace models to ONNX format."""

    def __init__(
        self,
        config: Optional[OnnxExportConfig] = None,
        memory_manager: Optional[MemoryManager] = None,
    ):
        self.config = config or OnnxExportConfig()
        self.memory_manager = memory_manager or MemoryManager()

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

        # Check memory requirements before attempting export
        memory_info = self.memory_manager.get_memory_info()
        available_mb = memory_info["available_mb"]

        # Estimate model requirements - for 1.1B models (~2.2GB), we need ~6-8GB for ONNX export
        model_size_mb = model_artifacts.get("size_mb", 1000.0)
        estimated_memory_mb = estimate_model_memory_mb(model_size_mb)

        # Hard memory limit check - if we clearly don't have enough, fail early
        if (
            available_mb > 0 and estimated_memory_mb > available_mb * 1.2
        ):  # Allow some overhead
            raise RuntimeError(
                f"Model too large for available memory:\n"
                f"  • Estimated requirement: {estimated_memory_mb:.1f}MB\n"
                f"  • Available memory: {available_mb:.1f}MB\n"
                f"  • Model: {model_id}\n\n"
                f"💡 Suggestions:\n"
                f"  • Try a smaller model\n"
                f"  • Increase system memory\n"
                f"  • Use a cloud instance with more RAM"
            )

        # Enable low memory mode if configured or if estimated memory exceeds threshold
        use_low_memory = self.config.low_memory
        if not use_low_memory and estimated_memory_mb > self.config.memory_threshold_mb:
            use_low_memory = True
            logger.info(
                f"Auto-enabling low memory mode: estimated {estimated_memory_mb:.1f}MB > threshold {self.config.memory_threshold_mb}MB"
            )

        if use_low_memory:
            self.memory_manager.set_low_memory_mode()
            # Aggressive pre-export cleanup
            self.memory_manager.force_garbage_collection()

            # Final memory check after cleanup
            memory_info = self.memory_manager.get_memory_info()
            is_safe, message = self.memory_manager.check_memory_threshold(
                estimated_memory_mb * 0.7
            )  # Use 70% of estimate as buffer
            if not is_safe:
                logger.warning(f"Proceeding with risky memory situation: {message}")

        try:
            with memory_managed_operation(
                self.memory_manager, f"ONNX export of {model_id}", estimated_memory_mb
            ):
                # Prepare export parameters
                export_kwargs = {
                    "model_id": model_id,
                    "export": True,
                    "use_cache": self.config.use_cache,
                    "provider": ExecutionProviders.CPU,  # Only CPU supported now
                }

                # Note: low_cpu_mem_usage not supported by optimum ONNX export
                # Memory management handled via garbage collection and environment settings

                # Use optimum for ONNX export - handles most transformer architectures
                ort_model = ORTModelForCausalLM.from_pretrained(**export_kwargs)

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
