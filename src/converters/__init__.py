"""
Alki Model Converter System

This module provides a pluggable architecture for converting models between
different formats (GGUF, ONNX, TensorRT, etc.). Each converter implements
the BaseConverter interface.

Use get_converter() to obtain the appropriate converter for a target format.
"""

from .base import BaseConverter, ConversionResult


def get_converter(target_format: str) -> BaseConverter:
    """
    Get converter for the specified target format.

    Args:
        target_format: Target format (e.g., 'gguf', 'onnx', 'tensorrt')

    Returns:
        Converter instance

    Raises:
        ValueError: If the target format is not supported
    """
    format_lower = target_format.lower()

    if format_lower == "gguf":
        from .gguf import GGUFConverter

        return GGUFConverter()
    elif format_lower == "onnx":
        # Future: from .onnx import ONNXConverter
        # return ONNXConverter()
        raise ValueError("ONNX conversion coming in Phase 2")
    elif format_lower == "tensorrt":
        # Future: from .tensorrt import TensorRTConverter
        # return TensorRTConverter()
        raise ValueError("TensorRT conversion coming in Phase 2")
    else:
        supported = get_supported_formats()
        raise ValueError(
            f"Unsupported format: {target_format}. Supported formats: {', '.join(supported)}"
        )


def get_supported_formats() -> list[str]:
    """
    Get list of all supported target formats.

    Returns:
        List of supported format names
    """
    return ["gguf"]  # Will expand to ["gguf", "onnx", "tensorrt"] in Phase 2


def list_converters() -> dict[str, dict]:
    """
    Get information about available converters.

    Returns:
        Dictionary mapping format names to converter details
    """
    result = {}

    for format_name in get_supported_formats():
        try:
            converter = get_converter(format_name)
            result[format_name] = {
                "name": converter.name,
                "target_format": converter.target_format,
                "supported_architectures": converter.supported_architectures,
                "supported_quantizations": converter.supported_quantizations,
            }
        except ValueError:
            # Skip formats that aren't implemented yet
            continue

    return result


__all__ = [
    "BaseConverter",
    "ConversionResult",
    "get_converter",
    "get_supported_formats",
    "list_converters",
]
