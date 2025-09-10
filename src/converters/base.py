"""
Base classes for the Alki converter plugin system.

This module defines the abstract interfaces that all converters must implement
to be compatible with the Alki plugin architecture.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional


@dataclass
class ConversionResult:
    """Result of a model conversion operation"""

    success: bool
    output_files: List[Path]
    source_model: str
    target_format: str
    quantization_profiles: List[str]
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseConverter(ABC):
    """
    Abstract base class for all model converters.

    Each converter implementation handles conversion from a source format
    (e.g., HuggingFace PyTorch) to a target format (e.g., GGUF, ONNX).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this converter"""
        pass

    @property
    @abstractmethod
    def target_format(self) -> str:
        """Target format this converter produces (e.g., 'gguf', 'onnx')"""
        pass

    @property
    @abstractmethod
    def supported_architectures(self) -> List[str]:
        """List of model architectures this converter supports"""
        pass

    @property
    @abstractmethod
    def supported_quantizations(self) -> List[str]:
        """List of quantization profiles this converter supports"""
        pass

    @abstractmethod
    def can_convert(self, source: str, architecture: Optional[str] = None) -> bool:
        """
        Check if this converter can handle the given source model.

        Args:
            source: Source model identifier (HF repo, local path, etc.)
            architecture: Model architecture if known

        Returns:
            True if this converter can handle the source
        """
        pass

    @abstractmethod
    def convert(
        self,
        source: str,
        output_dir: Path,
        quantizations: Optional[List[str]] = None,
        **kwargs,
    ) -> ConversionResult:
        """
        Convert a model from source format to target format.

        Args:
            source: Source model identifier (HF repo, local path, etc.)
            output_dir: Directory to write converted models
            quantizations: List of quantization profiles to apply
            **kwargs: Converter-specific options

        Returns:
            ConversionResult with success status and output files
        """
        pass

    @abstractmethod
    def validate_output(self, output_file: Path) -> bool:
        """
        Validate that the converted model is correct.

        Args:
            output_file: Path to converted model file

        Returns:
            True if the model is valid
        """
        pass

    def cleanup_intermediate_files(self, temp_dir: Path) -> None:
        """
        Clean up any temporary files created during conversion.

        Args:
            temp_dir: Directory containing temporary files
        """
        import shutil

        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    def get_conversion_config(self) -> Dict[str, Any]:
        """
        Get converter-specific configuration options.

        Returns:
            Dictionary of configuration options with descriptions
        """
        return {}
