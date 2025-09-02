from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional

import onnx
from optimum.onnxruntime import ORTModelForCausalLM


@dataclass
class OnnxExportConfig:
    """Configuration for ONNX export."""

    opset_version: int = 14
    use_gpu: bool = False
    optimize: bool = True
    output_dir: Optional[Path] = None

    # Dynamic axes for variable sequence lengths
    dynamic_axes: Dict[str, Dict[int, str]] = None

    def __post_init__(self):
        if self.dynamic_axes is None:
            # Default dynamic axes for causal LM models
            self.dynamic_axes = {
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "past_key_values": {0: "batch_size", 2: "past_sequence_length"},
            }


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
                use_cache=True,
                provider=(
                    "CPUExecutionProvider"
                    if not self.config.use_gpu
                    else "CUDAExecutionProvider"
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

            # Check for common unsupported architectures
            if any(
                unsupported in architecture.lower()
                for unsupported in ["mamba", "mixtral", "phi3", "qwen2"]
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
        # Known supported architectures
        supported = [
            "gpt2",
            "bert",
            "distilbert",
            "roberta",
            "xlm-roberta",
            "gpt-neo",
            "gpt-j",
            "opt",
            "bloom",
            "t5",
            "bart",
            "llama",
            "mistral",
            "codegen",
            "falcon",
        ]

        # Known problematic architectures
        unsupported = ["mamba", "mixtral", "phi3", "qwen2", "gemma"]

        arch_lower = architecture.lower()

        if any(arch in arch_lower for arch in unsupported):
            return False

        return any(arch in arch_lower for arch in supported)
