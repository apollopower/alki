"""
Bundle Builder for Alki Edge Deployment

This module implements the BundleBuilder class that orchestrates the creation
of deployment bundles from the outputs of the Alki pipeline:
- Model loading artifacts
- ONNX export results
- Quantization outputs

The builder organizes all artifacts into the standard bundle directory
structure and generates the bundle.yaml manifest.
"""

import shutil
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import logging

from .bundle import (
    Bundle,
    BundleMetadata,
    RuntimeConfig,
    BundleArtifacts,
    create_bundle_directory_structure,
)
from .constants import (
    FileNames,
    ExecutionProviders,
    QuantizationMethods,
    QuantizationFormats,
    Targets,
    FileSizes,
)

logger = logging.getLogger(__name__)


class BundleBuilder:
    """
    Orchestrates the creation of deployment bundles from pipeline artifacts.

    Takes outputs from HuggingFaceModelLoader, OnnxExporter, and SmoothQuantizer
    and packages them into a self-contained bundle ready for edge deployment.
    """

    def __init__(self, alki_version: str = "0.1.0"):
        """
        Initialize the bundle builder.

        Args:
            alki_version: Version of Alki creating this bundle
        """
        self.alki_version = alki_version

    def build_bundle(
        self,
        model_artifacts: Dict[str, Any],
        onnx_artifacts: Dict[str, Any],
        quantization_artifacts: Optional[Dict[str, Any]],
        output_path: Path,
        target: str = "cpu",
        preset: str = "balanced",
    ) -> Bundle:
        """
        Build a complete deployment bundle from pipeline artifacts.

        Args:
            model_artifacts: Output from HuggingFaceModelLoader.prepare()
            onnx_artifacts: Output from OnnxExporter.export()
            quantization_artifacts: Output from SmoothQuantizer.quantize_model() (optional)
            output_path: Directory where bundle should be created
            target: Target deployment type (e.g., "cpu", "openvino")
            preset: Optimization preset used (e.g., "balanced", "fast", "small")

        Returns:
            Bundle object representing the created bundle
        """
        logger.info(
            f"Building bundle for {model_artifacts['model_id']} targeting {target}"
        )

        # Create bundle directory structure
        create_bundle_directory_structure(output_path)

        # Create bundle metadata
        metadata = self._create_metadata(
            model_artifacts, onnx_artifacts, quantization_artifacts, target, preset
        )

        # Create runtime configuration
        runtime_config = self._create_runtime_config(
            onnx_artifacts, quantization_artifacts, target
        )

        # Copy and organize artifacts
        artifacts = self._organize_artifacts(
            model_artifacts, onnx_artifacts, quantization_artifacts, output_path
        )

        # Create bundle object
        bundle = Bundle(
            metadata=metadata,
            runtime_config=runtime_config,
            artifacts=artifacts,
            bundle_path=output_path,
        )

        bundle_yaml_path = output_path / FileNames.BUNDLE_MANIFEST
        bundle.save_yaml(bundle_yaml_path)

        logger.info(f"✓ Bundle created successfully at: {output_path}")
        logger.info(f"  Bundle manifest: {bundle_yaml_path}")

        # Log bundle statistics
        self._log_bundle_statistics(bundle)

        return bundle

    def _create_metadata(
        self,
        model_artifacts: Dict[str, Any],
        onnx_artifacts: Dict[str, Any],
        quantization_artifacts: Optional[Dict[str, Any]],
        target: str,
        preset: str,
    ) -> BundleMetadata:
        """Create bundle metadata from pipeline artifacts."""

        size_info = self._calculate_size_info(onnx_artifacts, quantization_artifacts)

        quant_info = self._extract_quantization_info(quantization_artifacts)

        return BundleMetadata(
            model_id=model_artifacts["model_id"],
            architecture=model_artifacts["architecture"],
            alki_version=self.alki_version,
            created_at=datetime.now(),
            target=target,
            preset=preset,
            **size_info,
            **quant_info,
        )

    def _create_runtime_config(
        self,
        onnx_artifacts: Dict[str, Any],
        quantization_artifacts: Optional[Dict[str, Any]],
        target: str,
    ) -> RuntimeConfig:
        """Create runtime configuration from pipeline artifacts."""

        # Extract configuration from ONNX export
        export_config = onnx_artifacts.get("export_config")

        provider = self._determine_execution_provider(target, export_config)

        model_info = self._extract_model_info(export_config, onnx_artifacts)

        quant_config = self._get_quantization_config(quantization_artifacts)

        return RuntimeConfig(
            provider=provider,
            **model_info,
            **quant_config,
        )

    def _organize_artifacts(
        self,
        model_artifacts: Dict[str, Any],
        onnx_artifacts: Dict[str, Any],
        quantization_artifacts: Optional[Dict[str, Any]],
        bundle_path: Path,
    ) -> BundleArtifacts:
        """Copy and organize all artifacts into bundle directory structure."""

        model_source_path = None
        model_dest_path = bundle_path / FileNames.ONNX_MODEL

        if quantization_artifacts and "quantized_model_path" in quantization_artifacts:
            # Use quantized model as main model
            model_source_path = Path(quantization_artifacts["quantized_model_path"])
        elif "onnx_path" in onnx_artifacts:
            # Use original ONNX model
            model_source_path = Path(onnx_artifacts["onnx_path"])

        if model_source_path and model_source_path.exists():
            shutil.copy2(model_source_path, model_dest_path)
            logger.info(f"Copied model: {model_source_path} -> {model_dest_path}")
        else:
            raise FileNotFoundError("No valid ONNX model found in artifacts")

        # Copy tokenizer files
        self._copy_tokenizer_artifacts(model_artifacts, bundle_path)

        artifacts = BundleArtifacts(
            model_onnx=FileNames.ONNX_MODEL,
            tokenizer_dir=FileNames.TOKENIZER_DIR,
            tokenizer_config=f"{FileNames.TOKENIZER_DIR}/{FileNames.TOKENIZER_CONFIG}",
            tokenizer_json=f"{FileNames.TOKENIZER_DIR}/{FileNames.TOKENIZER_JSON}",
            special_tokens_map=f"{FileNames.TOKENIZER_DIR}/{FileNames.TOKENIZER_SPECIAL_TOKENS}",
        )

        if (
            quantization_artifacts
            and "quantized_model_path" in quantization_artifacts
            and "onnx_path" in onnx_artifacts
        ):
            original_path = bundle_path / FileNames.ONNX_MODEL_ORIGINAL
            shutil.copy2(Path(onnx_artifacts["onnx_path"]), original_path)
            artifacts.model_original = FileNames.ONNX_MODEL_ORIGINAL

        return artifacts

    def _copy_tokenizer_artifacts(
        self, model_artifacts: Dict[str, Any], bundle_path: Path
    ) -> None:
        """Copy tokenizer files to bundle tokenizer directory."""

        tokenizer_dest_dir = bundle_path / FileNames.TOKENIZER_DIR
        tokenizer_dest_dir.mkdir(exist_ok=True)

        # Get tokenizer from model artifacts
        tokenizer = model_artifacts.get("tokenizer")
        if not tokenizer:
            logger.warning("No tokenizer found in model artifacts")
            return

        # Get the tokenizer's source directory (where HF downloaded it)
        local_path = model_artifacts.get("local_path")
        if not local_path or not Path(local_path).exists():
            logger.warning("No local model path found for tokenizer files")
            return

        tokenizer_source_dir = Path(local_path)

        tokenizer_files = [
            FileNames.TOKENIZER_CONFIG,
            FileNames.TOKENIZER_JSON,
            FileNames.TOKENIZER_VOCAB,
            FileNames.TOKENIZER_MERGES,
            FileNames.TOKENIZER_SPECIAL_TOKENS,
            FileNames.TOKENIZER_VOCAB_TXT,
        ]

        copied_files = []
        for filename in tokenizer_files:
            source_path = tokenizer_source_dir / filename
            if source_path.exists():
                dest_path = tokenizer_dest_dir / filename
                shutil.copy2(source_path, dest_path)
                copied_files.append(filename)
                logger.debug(f"Copied tokenizer file: {filename}")

        if not copied_files:
            # Fallback: save tokenizer using transformers save_pretrained
            try:
                tokenizer.save_pretrained(tokenizer_dest_dir)
                logger.info(f"Saved tokenizer to: {tokenizer_dest_dir}")
            except Exception as e:
                logger.error(f"Failed to save tokenizer: {e}")
                raise
        else:
            logger.info(f"Copied {len(copied_files)} tokenizer files: {copied_files}")

    def _log_bundle_statistics(self, bundle: Bundle) -> None:
        """Log bundle creation statistics."""

        metadata = bundle.metadata

        # Size information
        if metadata.original_size_mb and metadata.quantized_size_mb:
            logger.info(
                f"  Model size: {metadata.original_size_mb}MB → {metadata.quantized_size_mb}MB "
                f"({100 * (1 - metadata.compression_ratio):.1f}% reduction)"
            )
        elif metadata.original_size_mb:
            logger.info(f"  Model size: {metadata.original_size_mb}MB (unquantized)")

        # Quantization info
        if metadata.quantization_method:
            logger.info(f"  Quantization: {metadata.quantization_method}")
            if metadata.quantization_alpha is not None:
                logger.info(f"  Alpha parameter: {metadata.quantization_alpha}")

        # Runtime info
        logger.info(f"  Target: {metadata.target}")
        logger.info(f"  Provider: {bundle.runtime_config.provider}")

        # Validation
        issues = bundle.validate()
        if issues:
            logger.warning(f"Bundle validation found {len(issues)} issues:")
            for issue in issues:
                logger.warning(f"    - {issue}")
        else:
            logger.info("  Bundle validation: ✓ Passed")

    def _calculate_size_info(
        self,
        onnx_artifacts: Dict[str, Any],
        quantization_artifacts: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Calculate model size information."""
        original_size_mb = None
        quantized_size_mb = None
        compression_ratio = None

        if "onnx_path" in onnx_artifacts:
            onnx_path = Path(onnx_artifacts["onnx_path"])
            if onnx_path.exists():
                original_size_mb = onnx_path.stat().st_size // FileSizes.MB

        if quantization_artifacts and "quantized_model_path" in quantization_artifacts:
            quant_path = Path(quantization_artifacts["quantized_model_path"])
            if quant_path.exists():
                quantized_size_mb = quant_path.stat().st_size // FileSizes.MB
                if original_size_mb and original_size_mb > 0:
                    compression_ratio = quantized_size_mb / original_size_mb

        return {
            "original_size_mb": original_size_mb,
            "quantized_size_mb": quantized_size_mb,
            "compression_ratio": compression_ratio,
        }

    def _extract_quantization_info(
        self, quantization_artifacts: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract quantization method and parameters."""
        quantization_method = None
        quantization_alpha = None

        if quantization_artifacts and "config" in quantization_artifacts:
            config = quantization_artifacts["config"]
            quantization_method = QuantizationMethods.SMOOTHQUANT_W8A8
            if hasattr(config, "alpha"):
                quantization_alpha = config.alpha

        return {
            "quantization_method": quantization_method,
            "quantization_alpha": quantization_alpha,
        }

    def _determine_execution_provider(self, target: str, export_config) -> str:
        """Determine the appropriate execution provider for the target."""
        if target == Targets.OPENVINO:
            return ExecutionProviders.OPENVINO
        elif (
            export_config
            and hasattr(export_config, "use_gpu")
            and export_config.use_gpu
        ):
            return ExecutionProviders.CUDA
        else:
            return ExecutionProviders.CPU

    def _extract_model_info(
        self, export_config, onnx_artifacts: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract ONNX model configuration information."""
        from .constants import Defaults, ModelIO

        opset_version = Defaults.ONNX_OPSET_VERSION
        use_cache = False
        input_names = ModelIO.DEFAULT_INPUTS
        output_names = ModelIO.DEFAULT_OUTPUTS

        if export_config:
            if hasattr(export_config, "opset_version"):
                opset_version = export_config.opset_version
            if hasattr(export_config, "use_cache"):
                use_cache = export_config.use_cache

        if "onnx_model" in onnx_artifacts:
            try:
                onnx_model = onnx_artifacts["onnx_model"]
                if hasattr(onnx_model, "graph"):
                    input_names = [inp.name for inp in onnx_model.graph.input]
                    output_names = [out.name for out in onnx_model.graph.output]
            except Exception as e:
                logger.debug(f"Could not extract I/O names from ONNX model: {e}")

        return {
            "opset_version": opset_version,
            "use_cache": use_cache,
            "input_names": input_names,
            "output_names": output_names,
        }

    def _get_quantization_config(
        self, quantization_artifacts: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Get quantization configuration for runtime."""
        is_quantized = quantization_artifacts is not None
        quantization_format = None
        activation_type = None
        weight_type = None

        if quantization_artifacts and "config" in quantization_artifacts:
            quantization_format = QuantizationFormats.QDQ
            activation_type = QuantizationFormats.QINT8
            weight_type = QuantizationFormats.QINT8

        return {
            "is_quantized": is_quantized,
            "quantization_format": quantization_format,
            "activation_type": activation_type,
            "weight_type": weight_type,
        }


def create_bundle_from_pipeline(
    model_artifacts: Dict[str, Any],
    onnx_artifacts: Dict[str, Any],
    quantization_artifacts: Optional[Dict[str, Any]],
    output_path: Path,
    target: str = "cpu",
    preset: str = "balanced",
) -> Bundle:
    """
    Convenience function to create a bundle from pipeline artifacts.

    This is the main entry point for creating bundles from the Alki pipeline.

    Args:
        model_artifacts: Output from HuggingFaceModelLoader.prepare()
        onnx_artifacts: Output from OnnxExporter.export()
        quantization_artifacts: Output from SmoothQuantizer.quantize_model()
        output_path: Where to create the bundle
        target: Target deployment type
        preset: Optimization preset

    Returns:
        Bundle object representing the created bundle
    """
    builder = BundleBuilder()
    return builder.build_bundle(
        model_artifacts=model_artifacts,
        onnx_artifacts=onnx_artifacts,
        quantization_artifacts=quantization_artifacts,
        output_path=output_path,
        target=target,
        preset=preset,
    )
