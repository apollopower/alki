"""Unit tests for Bundle data model and serialization."""

import pytest
import tempfile
import yaml
from pathlib import Path
from datetime import datetime

from src.core.bundle import (
    Bundle,
    BundleMetadata,
    RuntimeConfig,
    BundleArtifacts,
    create_bundle_directory_structure,
)


class TestBundleMetadata:
    """Test BundleMetadata dataclass and serialization."""

    def test_bundle_metadata_creation(self):
        """Test basic metadata creation."""
        metadata = BundleMetadata(
            model_id="gpt2",
            architecture="GPT2LMHeadModel",
            alki_version="0.1.0",
            created_at=datetime(2024, 1, 15, 12, 0, 0),
            target="cpu",
            preset="balanced",
        )

        assert metadata.model_id == "gpt2"
        assert metadata.architecture == "GPT2LMHeadModel"
        assert metadata.target == "cpu"
        assert metadata.preset == "balanced"

    def test_metadata_to_dict(self):
        """Test metadata serialization to dictionary."""
        metadata = BundleMetadata(
            model_id="gpt2",
            architecture="GPT2LMHeadModel",
            alki_version="0.1.0",
            created_at=datetime(2024, 1, 15, 12, 0, 0),
            target="cpu",
            preset="balanced",
            original_size_mb=100,
            quantized_size_mb=25,
            compression_ratio=0.25,
            quantization_method="SmoothQuant W8A8",
            quantization_alpha=0.5,
        )

        result = metadata.to_dict()

        assert result["model_id"] == "gpt2"
        assert result["created_at"] == "2024-01-15T12:00:00"
        assert result["original_size_mb"] == 100
        assert result["quantized_size_mb"] == 25
        assert result["compression_ratio"] == 0.25
        assert result["quantization_method"] == "SmoothQuant W8A8"
        assert result["quantization_alpha"] == 0.5

    def test_metadata_from_dict(self):
        """Test metadata deserialization from dictionary."""
        data = {
            "model_id": "gpt2",
            "architecture": "GPT2LMHeadModel",
            "alki_version": "0.1.0",
            "created_at": "2024-01-15T12:00:00",
            "target": "cpu",
            "preset": "balanced",
            "original_size_mb": 100,
            "quantization_method": "SmoothQuant W8A8",
        }

        metadata = BundleMetadata.from_dict(data)

        assert metadata.model_id == "gpt2"
        assert metadata.created_at == datetime(2024, 1, 15, 12, 0, 0)
        assert metadata.original_size_mb == 100
        assert metadata.quantization_method == "SmoothQuant W8A8"


class TestRuntimeConfig:
    """Test RuntimeConfig dataclass and serialization."""

    def test_runtime_config_defaults(self):
        """Test default runtime configuration."""
        config = RuntimeConfig()

        assert config.provider == "CPUExecutionProvider"
        assert config.opset_version == 14
        assert config.use_cache is False
        assert config.max_sequence_length == 512
        assert config.input_names == ["input_ids", "attention_mask"]
        assert config.output_names == ["logits"]
        assert config.is_quantized is False

    def test_runtime_config_custom_values(self):
        """Test custom runtime configuration."""
        config = RuntimeConfig(
            provider="CPUExecutionProvider",
            opset_version=16,
            use_cache=True,
            is_quantized=True,
            quantization_format="QDQ",
            activation_type="QInt8",
            weight_type="QInt8",
        )

        assert config.provider == "CPUExecutionProvider"
        assert config.opset_version == 16
        assert config.use_cache is True
        assert config.is_quantized is True
        assert config.quantization_format == "QDQ"

    def test_runtime_config_serialization(self):
        """Test runtime config to/from dict conversion."""
        config = RuntimeConfig(
            provider="CPUExecutionProvider",
            is_quantized=True,
            quantization_format="QDQ",
        )

        # Test to_dict
        result = config.to_dict()
        assert result["provider"] == "CPUExecutionProvider"
        assert result["is_quantized"] is True
        assert result["quantization_format"] == "QDQ"

        # Test from_dict
        restored = RuntimeConfig.from_dict(result)
        assert restored.provider == "CPUExecutionProvider"
        assert restored.is_quantized is True
        assert restored.quantization_format == "QDQ"


class TestBundleArtifacts:
    """Test BundleArtifacts dataclass and serialization."""

    def test_bundle_artifacts_defaults(self):
        """Test default artifact paths."""
        artifacts = BundleArtifacts()

        assert artifacts.model_onnx == "model.onnx"
        assert artifacts.tokenizer_dir == "tokenizer"
        assert artifacts.tokenizer_config == "tokenizer/tokenizer_config.json"
        assert artifacts.tokenizer_json == "tokenizer/tokenizer.json"
        assert artifacts.special_tokens_map == "tokenizer/special_tokens_map.json"

    def test_bundle_artifacts_custom_paths(self):
        """Test custom artifact paths."""
        artifacts = BundleArtifacts(
            model_onnx="custom_model.onnx",
            model_original="original_model.onnx",
            vocab_file="tokenizer/vocab.txt",
            merges_file="tokenizer/merges.txt",
        )

        assert artifacts.model_onnx == "custom_model.onnx"
        assert artifacts.model_original == "original_model.onnx"
        assert artifacts.vocab_file == "tokenizer/vocab.txt"
        assert artifacts.merges_file == "tokenizer/merges.txt"

    def test_artifacts_serialization(self):
        """Test artifacts to/from dict conversion."""
        artifacts = BundleArtifacts(
            model_original="original.onnx", vocab_file="tokenizer/vocab.txt"
        )

        # Test to_dict
        result = artifacts.to_dict()
        assert result["model_onnx"] == "model.onnx"
        assert result["model_original"] == "original.onnx"
        assert result["vocab_file"] == "tokenizer/vocab.txt"

        # Test from_dict
        restored = BundleArtifacts.from_dict(result)
        assert restored.model_original == "original.onnx"
        assert restored.vocab_file == "tokenizer/vocab.txt"


class TestBundle:
    """Test complete Bundle functionality."""

    @pytest.fixture
    def sample_bundle(self):
        """Create a sample bundle for testing."""
        metadata = BundleMetadata(
            model_id="gpt2",
            architecture="GPT2LMHeadModel",
            alki_version="0.1.0",
            created_at=datetime(2024, 1, 15, 12, 0, 0),
            target="cpu",
            preset="balanced",
        )

        runtime_config = RuntimeConfig(
            provider="CPUExecutionProvider", is_quantized=False
        )

        artifacts = BundleArtifacts()

        return Bundle(
            metadata=metadata, runtime_config=runtime_config, artifacts=artifacts
        )

    def test_bundle_creation(self, sample_bundle):
        """Test basic bundle creation."""
        assert sample_bundle.metadata.model_id == "gpt2"
        assert sample_bundle.runtime_config.provider == "CPUExecutionProvider"
        assert sample_bundle.artifacts.model_onnx == "model.onnx"
        assert sample_bundle.bundle_path is None

    def test_bundle_to_dict(self, sample_bundle):
        """Test bundle serialization to dictionary."""
        result = sample_bundle.to_dict()

        assert "metadata" in result
        assert "runtime" in result
        assert "artifacts" in result

        assert result["metadata"]["model_id"] == "gpt2"
        assert result["runtime"]["provider"] == "CPUExecutionProvider"
        assert result["artifacts"]["model_onnx"] == "model.onnx"

    def test_bundle_from_dict(self):
        """Test bundle deserialization from dictionary."""
        data = {
            "metadata": {
                "model_id": "gpt2",
                "architecture": "GPT2LMHeadModel",
                "alki_version": "0.1.0",
                "created_at": "2024-01-15T12:00:00",
                "target": "cpu",
                "preset": "balanced",
            },
            "runtime": {
                "provider": "CPUExecutionProvider",
                "opset_version": 14,
                "use_cache": False,
                "max_sequence_length": 512,
                "input_names": ["input_ids", "attention_mask"],
                "output_names": ["logits"],
                "is_quantized": False,
            },
            "artifacts": {
                "model_onnx": "model.onnx",
                "tokenizer_dir": "tokenizer",
                "tokenizer_config": "tokenizer/tokenizer_config.json",
            },
        }

        bundle = Bundle.from_dict(data)

        assert bundle.metadata.model_id == "gpt2"
        assert bundle.runtime_config.provider == "CPUExecutionProvider"
        assert bundle.artifacts.model_onnx == "model.onnx"

    def test_bundle_yaml_save_load(self, sample_bundle):
        """Test saving and loading bundle to/from YAML."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yaml_path = Path(temp_dir) / "bundle.yaml"

            # Save bundle
            sample_bundle.save_yaml(yaml_path)
            assert yaml_path.exists()

            # Verify YAML content
            with open(yaml_path) as f:
                yaml_data = yaml.safe_load(f)
            assert yaml_data["metadata"]["model_id"] == "gpt2"

            # Load bundle
            loaded_bundle = Bundle.load_yaml(yaml_path)
            assert loaded_bundle.metadata.model_id == "gpt2"
            assert loaded_bundle.bundle_path == yaml_path.parent

    def test_bundle_get_paths(self, sample_bundle):
        """Test bundle path getter methods."""
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle_path = Path(temp_dir)
            sample_bundle.bundle_path = bundle_path

            model_path = sample_bundle.get_model_path()
            tokenizer_path = sample_bundle.get_tokenizer_path()

            assert model_path == bundle_path / "model.onnx"
            assert tokenizer_path == bundle_path / "tokenizer"

    def test_bundle_get_paths_no_bundle_path(self, sample_bundle):
        """Test path getters raise error when bundle_path not set."""
        with pytest.raises(ValueError, match="Bundle path not set"):
            sample_bundle.get_model_path()

        with pytest.raises(ValueError, match="Bundle path not set"):
            sample_bundle.get_tokenizer_path()

    def test_bundle_validate_missing_files(self, sample_bundle):
        """Test bundle validation with missing files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle_path = Path(temp_dir)
            sample_bundle.bundle_path = bundle_path

            # Validation should find missing files
            issues = sample_bundle.validate()
            assert len(issues) > 0
            assert any("model file missing" in issue for issue in issues)
            assert any("Tokenizer directory missing" in issue for issue in issues)

    def test_bundle_validate_valid_bundle(self, sample_bundle):
        """Test bundle validation with all files present."""
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle_path = Path(temp_dir)
            sample_bundle.bundle_path = bundle_path

            # Create required files and directories
            create_bundle_directory_structure(bundle_path)

            # Create model file
            (bundle_path / "model.onnx").touch()

            # Create tokenizer config
            (bundle_path / "tokenizer" / "tokenizer_config.json").touch()

            # Validation should pass
            issues = sample_bundle.validate()
            assert len(issues) == 0
            assert sample_bundle.is_valid()

    def test_bundle_validate_quantization_mismatch(self, sample_bundle):
        """Test validation catches quantization metadata mismatch."""
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle_path = Path(temp_dir)
            sample_bundle.bundle_path = bundle_path

            # Set runtime as quantized but no metadata
            sample_bundle.runtime_config.is_quantized = True
            sample_bundle.metadata.quantization_method = None

            # Create required files to avoid other validation errors
            create_bundle_directory_structure(bundle_path)
            (bundle_path / "model.onnx").touch()
            (bundle_path / "tokenizer" / "tokenizer_config.json").touch()

            issues = sample_bundle.validate()
            assert len(issues) == 1
            assert "quantization method" in issues[0]


class TestBundleDirectoryStructure:
    """Test bundle directory creation utilities."""

    def test_create_bundle_directory_structure(self):
        """Test creating standard bundle directory structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle_path = Path(temp_dir) / "test_bundle"

            create_bundle_directory_structure(bundle_path)

            assert bundle_path.exists()
            assert (bundle_path / "tokenizer").exists()
            assert (bundle_path / "runners").exists()
            assert bundle_path.is_dir()
            assert (bundle_path / "tokenizer").is_dir()
            assert (bundle_path / "runners").is_dir()

    def test_create_bundle_directory_structure_exists(self):
        """Test creating directory structure when it already exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle_path = Path(temp_dir)

            # Create directories first
            (bundle_path / "tokenizer").mkdir()

            # Should not raise error
            create_bundle_directory_structure(bundle_path)

            assert bundle_path.exists()
            assert (bundle_path / "tokenizer").exists()
            assert (bundle_path / "runners").exists()
