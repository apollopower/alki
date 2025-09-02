"""
Bundle Manager for Alki Edge Deployment

This module provides functionality for loading, validating, and managing
existing deployment bundles. It handles bundle discovery, loading from
various sources, and runtime compatibility checks.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging

from .bundle import Bundle

logger = logging.getLogger(__name__)


class BundleManager:
    """
    Manages loading, validation, and inspection of deployment bundles.

    Provides high-level operations for working with bundles created by
    the BundleBuilder, including loading from disk, validating integrity,
    and checking runtime compatibility.
    """

    def __init__(self):
        """Initialize the bundle manager."""
        pass

    def load_bundle(self, bundle_path: Union[str, Path]) -> Bundle:
        """
        Load a bundle from a directory or bundle.yaml file.

        Args:
            bundle_path: Path to bundle directory or bundle.yaml file

        Returns:
            Loaded and validated Bundle object

        Raises:
            FileNotFoundError: If bundle path doesn't exist
            ValueError: If bundle is invalid or corrupted
        """
        bundle_path = Path(bundle_path)

        # Determine if path is directory or YAML file
        if bundle_path.is_file() and bundle_path.name == "bundle.yaml":
            yaml_path = bundle_path
        elif bundle_path.is_dir():
            yaml_path = bundle_path / "bundle.yaml"
        else:
            raise FileNotFoundError(f"Bundle not found at: {bundle_path}")

        logger.info(f"Loading bundle from: {yaml_path.parent}")

        # Load bundle from YAML
        try:
            bundle = Bundle.load_yaml(yaml_path)
        except Exception as e:
            raise ValueError(f"Failed to load bundle configuration: {e}") from e

        # Validate bundle integrity
        validation_issues = bundle.validate()
        if validation_issues:
            error_msg = (
                f"Bundle validation failed with {len(validation_issues)} issues:\n"
            )
            for issue in validation_issues:
                error_msg += f"  - {issue}\n"
            raise ValueError(error_msg.strip())

        logger.info(f"✓ Successfully loaded bundle: {bundle.metadata.model_id}")
        return bundle

    def discover_bundles(
        self, search_path: Path, recursive: bool = True
    ) -> List[Bundle]:
        """
        Discover all valid bundles in a directory tree.

        Args:
            search_path: Directory to search for bundles
            recursive: Whether to search subdirectories recursively

        Returns:
            List of discovered and validated Bundle objects
        """
        bundles = []

        if not search_path.exists():
            logger.warning(f"Search path does not exist: {search_path}")
            return bundles

        logger.info(f"Searching for bundles in: {search_path}")

        # Find all bundle.yaml files
        if recursive:
            pattern = "**/bundle.yaml"
        else:
            # For non-recursive, look in immediate subdirectories only
            pattern = "*/bundle.yaml"
        yaml_files = list(search_path.glob(pattern))

        logger.info(f"Found {len(yaml_files)} bundle.yaml files")

        for yaml_path in yaml_files:
            try:
                bundle = self.load_bundle(yaml_path)
                bundles.append(bundle)
                logger.debug(f"Discovered bundle: {bundle.metadata.model_id}")
            except Exception as e:
                logger.warning(f"Skipping invalid bundle at {yaml_path}: {e}")

        logger.info(f"Successfully discovered {len(bundles)} valid bundles")
        return bundles

    def list_bundle_info(self, bundle: Bundle) -> Dict[str, any]:
        """
        Extract key information from a bundle for display/inspection.

        Args:
            bundle: Bundle to inspect

        Returns:
            Dictionary with bundle information suitable for display
        """
        metadata = bundle.metadata
        runtime = bundle.runtime_config

        info = {
            "model_id": metadata.model_id,
            "architecture": metadata.architecture,
            "target": metadata.target,
            "preset": metadata.preset,
            "created_at": metadata.created_at.isoformat(),
            "alki_version": metadata.alki_version,
            "is_quantized": runtime.is_quantized,
            "provider": runtime.provider,
            "bundle_path": str(bundle.bundle_path) if bundle.bundle_path else None,
        }

        # Add size information if available
        if metadata.original_size_mb:
            info["original_size_mb"] = metadata.original_size_mb
        if metadata.quantized_size_mb:
            info["quantized_size_mb"] = metadata.quantized_size_mb
        if metadata.compression_ratio:
            info["compression_ratio"] = metadata.compression_ratio
            info["size_reduction_percent"] = round(
                100 * (1 - metadata.compression_ratio), 1
            )

        # Add quantization details if available
        if metadata.quantization_method:
            info["quantization_method"] = metadata.quantization_method
        if metadata.quantization_alpha is not None:
            info["quantization_alpha"] = metadata.quantization_alpha

        return info

    def validate_runtime_compatibility(
        self, bundle: Bundle, target_provider: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Check if bundle is compatible with current runtime environment.

        Args:
            bundle: Bundle to check compatibility for
            target_provider: Specific provider to check (optional)

        Returns:
            Dictionary with compatibility information and any issues
        """
        runtime = bundle.runtime_config
        compatibility = {
            "compatible": True,
            "issues": [],
            "recommendations": [],
        }

        # Check execution provider compatibility
        required_provider = runtime.provider
        if target_provider and target_provider != required_provider:
            compatibility["compatible"] = False
            compatibility["issues"].append(
                f"Bundle requires {required_provider} but {target_provider} requested"
            )

        # Check for provider availability (simplified check)
        available_providers = self._get_available_providers()
        if required_provider not in available_providers:
            compatibility["compatible"] = False
            compatibility["issues"].append(
                f"Required provider {required_provider} not available. "
                f"Available: {', '.join(available_providers)}"
            )

        # Check quantization support
        if runtime.is_quantized:
            if required_provider == "CPUExecutionProvider":
                # CPU provider generally supports quantization
                pass
            elif required_provider == "OpenVINOExecutionProvider":
                # OpenVINO supports quantization well
                pass
            elif required_provider == "CUDAExecutionProvider":
                # CUDA support for quantization varies
                compatibility["recommendations"].append(
                    "GPU quantization support may vary - test thoroughly"
                )

        # Check ONNX opset compatibility
        opset_version = runtime.opset_version
        if opset_version < 11:
            compatibility["recommendations"].append(
                f"ONNX opset {opset_version} is quite old - consider upgrading"
            )
        elif opset_version > 18:
            compatibility["recommendations"].append(
                f"ONNX opset {opset_version} is very new - ensure runtime support"
            )

        return compatibility

    def _get_available_providers(self) -> List[str]:
        """
        Get list of available execution providers on current system.

        Returns:
            List of available provider names
        """
        providers = ["CPUExecutionProvider"]  # Always available

        try:
            import onnxruntime as ort

            available = ort.get_available_providers()
            providers.extend([p for p in available if p not in providers])
        except ImportError:
            logger.warning("ONNX Runtime not available - limited provider detection")

        return providers

    def export_bundle_info(
        self, bundles: List[Bundle], output_path: Path, format: str = "json"
    ) -> None:
        """
        Export bundle information to file for external tools.

        Args:
            bundles: List of bundles to export info for
            output_path: Where to save the export file
            format: Export format ("json" or "yaml")
        """
        bundle_info_list = []

        for bundle in bundles:
            info = self.list_bundle_info(bundle)
            compatibility = self.validate_runtime_compatibility(bundle)
            info["compatibility"] = compatibility
            bundle_info_list.append(info)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format.lower() == "json":
            import json

            with open(output_path, "w") as f:
                json.dump(bundle_info_list, f, indent=2, default=str)
        elif format.lower() == "yaml":
            import yaml

            with open(output_path, "w") as f:
                yaml.safe_dump(bundle_info_list, f, indent=2, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported export format: {format}")

        logger.info(f"Exported info for {len(bundles)} bundles to: {output_path}")

    def get_bundle_stats(self, bundle: Bundle) -> Dict[str, any]:
        """
        Get detailed statistics about a bundle.

        Args:
            bundle: Bundle to analyze

        Returns:
            Dictionary with detailed bundle statistics
        """
        if not bundle.bundle_path:
            raise ValueError("Bundle path not set - cannot calculate statistics")

        stats = {
            "bundle_path": str(bundle.bundle_path),
            "total_files": 0,
            "total_size_bytes": 0,
            "file_breakdown": {},
        }

        # Count files and calculate sizes
        for root, dirs, files in os.walk(bundle.bundle_path):
            for file in files:
                file_path = Path(root) / file
                try:
                    size = file_path.stat().st_size
                    stats["total_files"] += 1
                    stats["total_size_bytes"] += size

                    # Categorize file
                    if file.endswith(".onnx"):
                        category = "models"
                    elif file_path.parent.name == "tokenizer":
                        category = "tokenizer"
                    elif file.endswith(".yaml"):
                        category = "config"
                    else:
                        category = "other"

                    if category not in stats["file_breakdown"]:
                        stats["file_breakdown"][category] = {
                            "count": 0,
                            "size_bytes": 0,
                        }

                    stats["file_breakdown"][category]["count"] += 1
                    stats["file_breakdown"][category]["size_bytes"] += size

                except (OSError, FileNotFoundError):
                    logger.debug(f"Could not stat file: {file_path}")

        # Convert to human readable sizes
        stats["total_size_mb"] = stats["total_size_bytes"] / (1024 * 1024)

        for category in stats["file_breakdown"]:
            size_bytes = stats["file_breakdown"][category]["size_bytes"]
            stats["file_breakdown"][category]["size_mb"] = size_bytes / (1024 * 1024)

        return stats


# Convenience functions for common operations


def load_bundle(bundle_path: Union[str, Path]) -> Bundle:
    """
    Convenience function to load a bundle.

    Args:
        bundle_path: Path to bundle directory or bundle.yaml file

    Returns:
        Loaded Bundle object
    """
    manager = BundleManager()
    return manager.load_bundle(bundle_path)


def discover_bundles(search_path: Path, recursive: bool = True) -> List[Bundle]:
    """
    Convenience function to discover bundles in a directory.

    Args:
        search_path: Directory to search
        recursive: Search subdirectories

    Returns:
        List of discovered Bundle objects
    """
    manager = BundleManager()
    return manager.discover_bundles(search_path, recursive)


def validate_bundle_compatibility(bundle: Bundle) -> Dict[str, any]:
    """
    Convenience function to check bundle compatibility.

    Args:
        bundle: Bundle to validate

    Returns:
        Compatibility information dictionary
    """
    manager = BundleManager()
    return manager.validate_runtime_compatibility(bundle)
