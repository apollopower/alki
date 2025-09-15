"""
GGUF converter implementation.

This converter handles conversion from HuggingFace models to GGUF format
using llama.cpp conversion and quantization tools.
"""

import logging
import shutil
from pathlib import Path
from typing import List, Optional

from huggingface_hub import snapshot_download, list_repo_files
from huggingface_hub.utils import RepositoryNotFoundError

from ..base import BaseConverter, ConversionResult
from ...core.tool_manager import ToolManager, DependencyError, ConversionError

logger = logging.getLogger(__name__)


class GGUFConverter(BaseConverter):
    """Converter for HuggingFace models to GGUF format"""

    def __init__(self):
        """Initialize the GGUF converter with tool manager."""
        self.tool_manager = ToolManager()

    @property
    def name(self) -> str:
        return "GGUF Converter (llama.cpp)"

    @property
    def target_format(self) -> str:
        return "gguf"

    @property
    def supported_architectures(self) -> List[str]:
        return [
            "LlamaForCausalLM",
            "QwenForCausalLM",
            "Qwen2ForCausalLM",
            "MistralForCausalLM",
            "PhiForCausalLM",
            "GemmaForCausalLM",
        ]

    @property
    def supported_quantizations(self) -> List[str]:
        return ["Q4_K_M", "Q5_K_M", "Q8_0", "Q4_0", "Q5_0", "Q6_K", "Q8_K"]

    def can_convert(self, source: str, architecture: Optional[str] = None) -> bool:
        """
        Check if this converter can handle the given source model.

        Args:
            source: HuggingFace repository ID or local path
            architecture: Model architecture if known

        Returns:
            True if this converter can handle the source
        """
        logger.debug(
            f"Checking if can convert: {source} (architecture: {architecture})"
        )

        # Check if dependencies are available
        missing_deps = self.tool_manager.check_dependencies()
        if missing_deps:
            logger.debug(f"Missing dependencies for conversion: {missing_deps}")
            return False

        # Check if Python environment is suitable
        if not self.tool_manager.check_python_environment():
            logger.debug("Python environment not suitable for conversion")
            return False

        # Check if source is a HuggingFace repository with PyTorch model files
        try:
            files = list_repo_files(source)

            # Must have config.json
            has_config = "config.json" in files

            # Must have model weights
            has_weights = any(f.endswith((".bin", ".safetensors")) for f in files)

            # Should not already be GGUF (use existing GGUF repos directly)
            has_gguf = any(f.endswith(".gguf") for f in files)

            can_convert = has_config and has_weights and not has_gguf

            logger.debug(
                f"Repository check - config: {has_config}, weights: {has_weights}, gguf: {has_gguf}"
            )

            return can_convert

        except RepositoryNotFoundError:
            logger.debug(f"Repository {source} not found")
            return False
        except Exception as e:
            logger.debug(f"Error checking repository {source}: {e}")
            return False

    def convert(
        self,
        source: str,
        output_dir: Path,
        quantizations: Optional[List[str]] = None,
        **kwargs,
    ) -> ConversionResult:
        """
        Convert HuggingFace model to GGUF format.

        Args:
            source: HuggingFace repository ID
            output_dir: Directory to write converted models
            quantizations: List of quantization profiles to apply
            **kwargs: Additional conversion options (timeout_minutes, cleanup)

        Returns:
            ConversionResult with success status and output files

        Raises:
            DependencyError: If required dependencies are missing
            ConversionError: If conversion fails
        """
        logger.info(f"Starting HuggingFace to GGUF conversion: {source}")

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Check dependencies first
        missing_deps = self.tool_manager.check_dependencies()
        if missing_deps:
            error_msg = (
                f"Missing required packages: {', '.join(missing_deps)}\n"
                f"Install with: pip install alki[convert]"
            )
            raise DependencyError(error_msg)

        # Set default quantization if none specified
        if not quantizations:
            quantizations = ["Q4_K_M"]

        output_files = []
        hf_model_path = None
        cleanup_hf_model = kwargs.get("cleanup", True)

        try:
            # Step 1: Download HuggingFace model
            logger.info(f"Downloading model from HuggingFace: {source}")
            hf_model_path = Path(snapshot_download(source))
            logger.debug(f"Model downloaded to: {hf_model_path}")

            # Step 2: Get conversion tool
            logger.info("Ensuring conversion tool is available...")
            script_path = self.tool_manager.ensure_conversion_tool()

            # Step 3: Convert for each quantization profile
            for i, quant in enumerate(quantizations):
                logger.info(
                    f"Converting with quantization {quant} ({i+1}/{len(quantizations)})"
                )

                # Generate output filename with conflict resolution
                model_name = source.split("/")[-1].lower().replace("-", "_")
                output_file = self._get_unique_output_path(
                    output_dir, model_name, quant
                )

                try:
                    # Convert to GGUF with specified quantization
                    success = self.tool_manager.run_conversion_script(
                        script_path=script_path,
                        hf_model_path=hf_model_path,
                        output_file=output_file,
                        outtype=self._map_quantization_type(quant),
                        timeout_minutes=kwargs.get("timeout_minutes", 60),
                    )

                    if success and self.validate_output(output_file):
                        output_files.append(output_file)
                        output_size_mb = output_file.stat().st_size / (1024 * 1024)
                        logger.info(
                            f"Successfully converted: {output_file} ({output_size_mb:.1f}MB)"
                        )
                    else:
                        logger.error(f"Validation failed for {output_file}")
                        # Continue with other quantizations instead of failing completely
                        continue

                except ConversionError as e:
                    logger.error(f"Failed to convert with quantization {quant}: {e}")
                    # Continue with other quantizations
                    continue

            # Check if any conversions succeeded
            if not output_files:
                raise ConversionError("All quantization conversions failed")

            logger.info(
                f"Conversion completed. Successfully created {len(output_files)}/{len(quantizations)} files."
            )

            return ConversionResult(
                success=True,
                output_files=output_files,
                source_model=source,
                target_format=self.target_format,
                quantization_profiles=[
                    q for i, q in enumerate(quantizations) if i < len(output_files)
                ],
            )

        except (DependencyError, ConversionError):
            # Clean up any partial output files on failure
            self._cleanup_partial_outputs(output_files)
            raise
        except Exception as e:
            logger.error(f"Unexpected error during conversion: {e}")
            self._cleanup_partial_outputs(output_files)
            raise ConversionError(f"Conversion failed: {str(e)}")
        finally:
            # Clean up downloaded HF model if requested
            if cleanup_hf_model and hf_model_path and hf_model_path.exists():
                self._cleanup_hf_model(hf_model_path)

    def _get_unique_output_path(
        self, output_dir: Path, model_name: str, quant: str
    ) -> Path:
        """Generate unique output path to avoid filename conflicts."""
        base_name = f"{model_name}_{quant.lower()}.gguf"
        output_file = output_dir / base_name

        # If file exists, add counter
        counter = 1
        while output_file.exists():
            name_with_counter = f"{model_name}_{quant.lower()}_{counter}.gguf"
            output_file = output_dir / name_with_counter
            counter += 1

        return output_file

    def _cleanup_partial_outputs(self, output_files: List[Path]) -> None:
        """Clean up partial output files on failure."""
        for output_file in output_files:
            try:
                if output_file.exists():
                    output_file.unlink()
                    logger.debug(f"Cleaned up partial file: {output_file}")
            except Exception as e:
                logger.warning(f"Could not clean up {output_file}: {e}")

    def _cleanup_hf_model(self, model_path: Path) -> None:
        """Clean up downloaded HuggingFace model directory."""
        try:
            # Only clean up if it looks like a HF cache directory
            if ".cache" in str(model_path) or "huggingface" in str(model_path):
                shutil.rmtree(model_path)
                logger.debug(f"Cleaned up HF model cache: {model_path}")
            else:
                logger.debug(f"Skipping cleanup of non-cache directory: {model_path}")
        except Exception as e:
            logger.warning(f"Could not clean up HF model directory {model_path}: {e}")

    def _map_quantization_type(self, quant: str) -> str:
        """
        Map Alki quantization names to llama.cpp quantization types.

        Args:
            quant: Alki quantization name (e.g., "Q4_K_M")

        Returns:
            llama.cpp quantization type
        """
        # Direct mapping for most cases
        mapping = {
            "Q4_K_M": "q4_k_m",
            "Q5_K_M": "q5_k_m",
            "Q8_0": "q8_0",
            "Q4_0": "q4_0",
            "Q5_0": "q5_0",
            "Q6_K": "q6_k",
            "Q8_K": "q8_k",
        }

        return mapping.get(quant, quant.lower())

    def validate_output(self, output_file: Path) -> bool:
        """
        Validate GGUF output file.

        Args:
            output_file: Path to GGUF file to validate

        Returns:
            True if file is valid
        """
        if not output_file.exists():
            logger.debug(f"Output file does not exist: {output_file}")
            return False

        if not output_file.suffix == ".gguf":
            logger.debug(f"Output file is not a GGUF file: {output_file}")
            return False

        # Check file is not empty
        if output_file.stat().st_size == 0:
            logger.debug(f"Output file is empty: {output_file}")
            return False

        # Basic GGUF header validation
        try:
            with open(output_file, "rb") as f:
                # GGUF files start with magic bytes "GGUF"
                magic = f.read(4)
                if magic != b"GGUF":
                    logger.debug(f"Invalid GGUF magic bytes in {output_file}")
                    return False

            logger.debug(f"GGUF file validation passed: {output_file}")
            return True

        except Exception as e:
            logger.debug(f"Error validating GGUF file {output_file}: {e}")
            return False
