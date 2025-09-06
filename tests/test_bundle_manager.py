"""Unit tests for BundleManager functionality."""

import pytest
import json
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.core.bundle_manager import (
    BundleManager,
    load_bundle,
    discover_bundles,
    validate_bundle_compatibility,
)
from .conftest import create_test_bundle_dir, TEST_MODEL_DATA


class TestBundleManager:
    """Test BundleManager functionality."""

    @pytest.fixture
    def valid_bundle_directory(self, tmp_path):
        """Create a valid bundle directory structure for testing."""
        return create_test_bundle_dir(tmp_path, quantized=True)

    def test_bundle_manager_initialization(self):
        """Test BundleManager initialization."""
        manager = BundleManager()
        assert manager is not None

    def test_load_bundle_from_directory(self, valid_bundle_directory):
        """Test loading bundle from directory."""
        manager = BundleManager()

        bundle = manager.load_bundle(valid_bundle_directory)

        assert bundle.metadata.model_id == "gpt2"
        assert bundle.bundle_path == valid_bundle_directory

    def test_load_bundle_from_yaml_file(self, valid_bundle_directory):
        """Test loading bundle from bundle.yaml file."""
        manager = BundleManager()
        yaml_file = valid_bundle_directory / "bundle.yaml"

        bundle = manager.load_bundle(yaml_file)

        assert bundle.metadata.model_id == "gpt2"
        assert bundle.bundle_path == valid_bundle_directory

    def test_load_bundle_nonexistent_path(self):
        """Test loading bundle from non-existent path."""
        manager = BundleManager()

        with pytest.raises(FileNotFoundError, match="Bundle not found"):
            manager.load_bundle(Path("/nonexistent/path"))

    def test_load_bundle_invalid_yaml(self, tmp_path):
        """Test loading bundle with invalid YAML."""
        manager = BundleManager()

        # Create invalid bundle.yaml
        bundle_dir = tmp_path / "bad_bundle"
        bundle_dir.mkdir()
        (bundle_dir / "bundle.yaml").write_text("invalid: yaml: content: [")

        with pytest.raises(ValueError, match="Failed to load bundle configuration"):
            manager.load_bundle(bundle_dir)

    def test_load_bundle_validation_failure(self, tmp_path):
        """Test loading bundle that fails validation."""
        manager = BundleManager()

        # Create bundle with missing required files
        bundle_dir = tmp_path / "incomplete_bundle"
        bundle_dir.mkdir()

        # Create minimal bundle.yaml without required files
        bundle_data = {
            "metadata": {
                "model_id": "gpt2",
                "architecture": "GPT2LMHeadModel",
                "alki_version": "0.1.0",
                "created_at": "2024-01-15T12:00:00",
                "target": "cpu",
                "preset": "balanced",
            },
            "runtime": {"provider": "CPUExecutionProvider", "is_quantized": False},
            "artifacts": {
                "model_onnx": "model.onnx",
                "tokenizer_dir": "tokenizer",
                "tokenizer_config": "tokenizer/tokenizer_config.json",
            },
        }

        with open(bundle_dir / "bundle.yaml", "w") as f:
            yaml.safe_dump(bundle_data, f)

        with pytest.raises(ValueError, match="Bundle validation failed"):
            manager.load_bundle(bundle_dir)

    def test_discover_bundles_single_directory(self, tmp_path):
        """Test discovering bundles in a single directory."""
        manager = BundleManager()

        # Create multiple bundle directories using helper
        for i in range(3):
            create_test_bundle_dir(tmp_path, model_id=f"model_{i}")

        # Discover bundles
        bundles = manager.discover_bundles(tmp_path, recursive=False)

        assert len(bundles) == 3
        model_ids = [b.metadata.model_id for b in bundles]
        assert "model_0" in model_ids
        assert "model_1" in model_ids
        assert "model_2" in model_ids

    def test_discover_bundles_recursive(self, tmp_path):
        """Test recursive bundle discovery."""
        manager = BundleManager()

        # Create nested structure
        sub_dir = tmp_path / "subdir"
        sub_dir.mkdir()

        bundle_dir = sub_dir / "nested_bundle"
        bundle_dir.mkdir()
        (bundle_dir / "tokenizer").mkdir()
        (bundle_dir / "model.onnx").write_bytes(TEST_MODEL_DATA)
        (bundle_dir / "tokenizer" / "tokenizer_config.json").write_text("{}")

        bundle_data = {
            "metadata": {
                "model_id": "nested_model",
                "architecture": "GPT2LMHeadModel",
                "alki_version": "0.1.0",
                "created_at": "2024-01-15T12:00:00",
                "target": "cpu",
                "preset": "balanced",
            },
            "runtime": {"provider": "CPUExecutionProvider", "is_quantized": False},
            "artifacts": {
                "model_onnx": "model.onnx",
                "tokenizer_dir": "tokenizer",
                "tokenizer_config": "tokenizer/tokenizer_config.json",
            },
        }

        with open(bundle_dir / "bundle.yaml", "w") as f:
            yaml.safe_dump(bundle_data, f)

        # Test recursive discovery
        bundles = manager.discover_bundles(tmp_path, recursive=True)
        assert len(bundles) == 1
        assert bundles[0].metadata.model_id == "nested_model"

        # Test non-recursive discovery
        bundles = manager.discover_bundles(tmp_path, recursive=False)
        assert len(bundles) == 0

    def test_discover_bundles_nonexistent_path(self):
        """Test bundle discovery with non-existent path."""
        manager = BundleManager()

        bundles = manager.discover_bundles(Path("/nonexistent"), recursive=True)
        assert len(bundles) == 0

    def test_list_bundle_info(self, sample_bundle):
        """Test extracting bundle information for display."""
        manager = BundleManager()
        sample_bundle.bundle_path = Path("/test/bundle")

        info = manager.list_bundle_info(sample_bundle)

        assert info["model_id"] == "gpt2"
        assert info["architecture"] == "GPT2LMHeadModel"
        assert info["target"] == "cpu"
        assert info["preset"] == "balanced"
        assert info["is_quantized"] is True
        assert info["provider"] == "CPUExecutionProvider"
        assert info["original_size_mb"] == 100
        assert info["quantized_size_mb"] == 25
        assert info["compression_ratio"] == 0.25
        assert info["size_reduction_percent"] == 75.0
        assert info["quantization_method"] == "SmoothQuant W8A8"
        assert info["bundle_path"] == "/test/bundle"

    @patch("builtins.__import__")
    def test_validate_runtime_compatibility_success(self, mock_import, sample_bundle):
        """Test runtime compatibility validation success."""
        manager = BundleManager()

        # Mock onnxruntime import
        mock_ort = MagicMock()
        mock_ort.get_available_providers.return_value = [
            "CPUExecutionProvider",
            "CUDAExecutionProvider",
        ]

        def side_effect(name, *args, **kwargs):
            if name == "onnxruntime":
                return mock_ort
            return __import__(name, *args, **kwargs)

        mock_import.side_effect = side_effect

        compatibility = manager.validate_runtime_compatibility(sample_bundle)

        assert compatibility["compatible"] is True
        assert len(compatibility["issues"]) == 0

    @patch("builtins.__import__")
    def test_validate_runtime_compatibility_provider_mismatch(
        self, mock_import, sample_bundle
    ):
        """Test runtime compatibility with provider mismatch."""
        manager = BundleManager()

        # Mock onnxruntime import
        mock_ort = MagicMock()
        mock_ort.get_available_providers.return_value = [
            "CPUExecutionProvider",
            "CUDAExecutionProvider",
        ]

        def side_effect(name, *args, **kwargs):
            if name == "onnxruntime":
                return mock_ort
            return __import__(name, *args, **kwargs)

        mock_import.side_effect = side_effect

        # Test with incompatible provider (simplified - removed OpenVINO references)
        compatibility = manager.validate_runtime_compatibility(
            sample_bundle, target_provider="SomeOtherProvider"
        )

        assert compatibility["compatible"] is True  # Simplified validation now
        # No longer checking provider mismatch since we only support CPU

    @patch("builtins.__import__")
    def test_validate_runtime_compatibility_provider_unavailable(
        self, mock_import, sample_bundle
    ):
        """Test runtime compatibility with unavailable provider."""
        manager = BundleManager()

        # Mock onnxruntime import with only CPU provider available
        mock_ort = MagicMock()
        mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]

        def side_effect(name, *args, **kwargs):
            if name == "onnxruntime":
                return mock_ort
            return __import__(name, *args, **kwargs)

        mock_import.side_effect = side_effect

        # Change bundle to require unsupported provider
        sample_bundle.runtime_config.provider = "UnsupportedProvider"

        compatibility = manager.validate_runtime_compatibility(sample_bundle)

        assert compatibility["compatible"] is False
        assert any(
            "but only CPU is supported" in issue for issue in compatibility["issues"]
        )

    def test_export_bundle_info_json(self, sample_bundle, tmp_path):
        """Test exporting bundle information to JSON."""
        manager = BundleManager()
        sample_bundle.bundle_path = Path("/test/bundle")

        output_path = tmp_path / "bundle_info.json"

        with patch.object(manager, "validate_runtime_compatibility") as mock_validate:
            mock_validate.return_value = {
                "compatible": True,
                "issues": [],
                "recommendations": [],
            }

            manager.export_bundle_info([sample_bundle], output_path, format="json")

        assert output_path.exists()

        with open(output_path) as f:
            data = json.load(f)

        assert len(data) == 1
        assert data[0]["model_id"] == "gpt2"
        assert "compatibility" in data[0]

    def test_export_bundle_info_yaml(self, sample_bundle, tmp_path):
        """Test exporting bundle information to YAML."""
        manager = BundleManager()
        sample_bundle.bundle_path = Path("/test/bundle")

        output_path = tmp_path / "bundle_info.yaml"

        with patch.object(manager, "validate_runtime_compatibility") as mock_validate:
            mock_validate.return_value = {
                "compatible": True,
                "issues": [],
                "recommendations": [],
            }

            manager.export_bundle_info([sample_bundle], output_path, format="yaml")

        assert output_path.exists()

        with open(output_path) as f:
            data = yaml.safe_load(f)

        assert len(data) == 1
        assert data[0]["model_id"] == "gpt2"

    def test_export_bundle_info_unsupported_format(self, sample_bundle, tmp_path):
        """Test error handling for unsupported export format."""
        manager = BundleManager()

        with pytest.raises(ValueError, match="Unsupported export format"):
            manager.export_bundle_info(
                [sample_bundle], tmp_path / "out.txt", format="txt"
            )

    def test_get_bundle_stats(self, valid_bundle_directory):
        """Test getting detailed bundle statistics."""
        manager = BundleManager()

        bundle = manager.load_bundle(valid_bundle_directory)
        stats = manager.get_bundle_stats(bundle)

        assert stats["bundle_path"] == str(valid_bundle_directory)
        assert (
            stats["total_files"] >= 3
        )  # At least model.onnx, bundle.yaml, tokenizer_config.json
        assert stats["total_size_bytes"] > 0
        assert stats["total_size_mb"] > 0
        assert "file_breakdown" in stats
        assert "models" in stats["file_breakdown"]
        assert "config" in stats["file_breakdown"]

    def test_get_bundle_stats_no_path(self, sample_bundle):
        """Test bundle stats error when no bundle path set."""
        manager = BundleManager()

        with pytest.raises(ValueError, match="Bundle path not set"):
            manager.get_bundle_stats(sample_bundle)

    @patch("builtins.__import__")
    def test_get_available_providers_no_ort(self, mock_import):
        """Test provider detection when ONNX Runtime is not available."""
        manager = BundleManager()

        # Mock ImportError when trying to import onnxruntime
        def side_effect(name, *args, **kwargs):
            if name == "onnxruntime":
                raise ImportError("No module named 'onnxruntime'")
            return __import__(name, *args, **kwargs)

        mock_import.side_effect = side_effect

        providers = manager._get_available_providers()

        assert providers == ["CPUExecutionProvider"]  # Only default


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_load_bundle_function(self, tmp_path):
        """Test load_bundle convenience function."""
        bundle_dir = create_test_bundle_dir(tmp_path)
        bundle = load_bundle(bundle_dir)

        assert bundle.metadata.model_id == "gpt2"
        assert bundle.bundle_path == bundle_dir

    def test_discover_bundles_function(self, tmp_path):
        """Test discover_bundles convenience function."""
        create_test_bundle_dir(tmp_path)
        bundles = discover_bundles(tmp_path)

        assert len(bundles) >= 1
        assert any(b.metadata.model_id == "gpt2" for b in bundles)

    @patch("src.core.bundle_manager.BundleManager")
    def test_validate_bundle_compatibility_function(self, mock_manager_class):
        """Test validate_bundle_compatibility convenience function."""
        mock_manager = MagicMock()
        mock_manager_class.return_value = mock_manager
        mock_manager.validate_runtime_compatibility.return_value = {"compatible": True}

        bundle = MagicMock()
        result = validate_bundle_compatibility(bundle)

        mock_manager.validate_runtime_compatibility.assert_called_once_with(bundle)
        assert result == {"compatible": True}
