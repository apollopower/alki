"""Unit tests for BundleBuilder functionality."""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime

from src.core.bundle_builder import BundleBuilder, create_bundle_from_pipeline
from src.core.bundle import Bundle
from .conftest import create_mock_onnx_model, create_mock_tokenizer, TEST_MODEL_DATA


class TestBundleBuilder:
    """Test BundleBuilder functionality."""

    @pytest.fixture
    def mock_model_artifacts(self):
        """Mock model artifacts from HuggingFaceModelLoader."""
        # Use common mock tokenizer helper
        mock_tokenizer = create_mock_tokenizer()

        return {
            "model_id": "gpt2",
            "architecture": "GPT2LMHeadModel",
            "local_path": "/mock/path/to/model",
            "tokenizer": mock_tokenizer,
            "config": MagicMock(),
        }

    @pytest.fixture
    def mock_onnx_artifacts(self, tmp_path):
        """Mock ONNX artifacts from OnnxExporter."""
        # Create a temporary ONNX file
        onnx_path = tmp_path / "model.onnx"
        onnx_path.write_bytes(TEST_MODEL_DATA)

        # Mock export config
        mock_config = MagicMock()
        mock_config.opset_version = 14
        mock_config.use_cache = False
        mock_config.use_gpu = False

        # Use common mock ONNX model helper
        mock_onnx_model = create_mock_onnx_model()

        return {
            "model_id": "gpt2",
            "architecture": "GPT2LMHeadModel",
            "onnx_path": onnx_path,
            "onnx_model": mock_onnx_model,
            "output_dir": tmp_path,
            "export_config": mock_config,
        }

    @pytest.fixture
    def mock_quantization_artifacts(self, tmp_path):
        """Mock quantization artifacts from SmoothQuantizer."""
        # Create a temporary quantized model file
        quant_path = tmp_path / "quantized.onnx"
        quant_path.write_bytes(TEST_MODEL_DATA)

        # Mock quantization config
        mock_config = MagicMock()
        mock_config.alpha = 0.5

        return {"quantized_model_path": quant_path, "config": mock_config}

    def test_bundle_builder_initialization(self):
        """Test BundleBuilder initialization."""
        builder = BundleBuilder()
        assert builder.alki_version == "0.1.0"

        builder_custom = BundleBuilder(alki_version="0.2.0")
        assert builder_custom.alki_version == "0.2.0"

    @patch("src.core.bundle_builder.shutil.copy2")
    def test_build_bundle_without_quantization(
        self, mock_copy, mock_model_artifacts, mock_onnx_artifacts, tmp_path
    ):
        """Test building bundle without quantization."""
        builder = BundleBuilder()
        output_path = tmp_path / "test_bundle"

        # Mock the tokenizer file copying behavior
        with patch.object(builder, "_copy_tokenizer_artifacts") as mock_copy_tokenizer:
            bundle = builder.build_bundle(
                model_artifacts=mock_model_artifacts,
                onnx_artifacts=mock_onnx_artifacts,
                quantization_artifacts=None,
                output_path=output_path,
                target="cpu",
                preset="balanced",
            )

        # Verify bundle structure
        assert isinstance(bundle, Bundle)
        assert bundle.metadata.model_id == "gpt2"
        assert bundle.metadata.target == "cpu"
        assert bundle.metadata.preset == "balanced"
        assert bundle.metadata.quantization_method is None
        assert bundle.runtime_config.provider == "CPUExecutionProvider"
        assert bundle.runtime_config.is_quantized is False
        assert bundle.bundle_path == output_path

        # Verify bundle.yaml was created
        assert (output_path / "bundle.yaml").exists()

        # Verify model was copied
        mock_copy.assert_called()
        mock_copy_tokenizer.assert_called_once()

    @patch("src.core.bundle_builder.shutil.copy2")
    def test_build_bundle_with_quantization(
        self,
        mock_copy,
        mock_model_artifacts,
        mock_onnx_artifacts,
        mock_quantization_artifacts,
        tmp_path,
    ):
        """Test building bundle with quantization."""
        builder = BundleBuilder()
        output_path = tmp_path / "test_bundle"

        with patch.object(builder, "_copy_tokenizer_artifacts"):
            bundle = builder.build_bundle(
                model_artifacts=mock_model_artifacts,
                onnx_artifacts=mock_onnx_artifacts,
                quantization_artifacts=mock_quantization_artifacts,
                output_path=output_path,
                target="cpu",
                preset="small",
            )

        # Verify quantization information
        assert bundle.metadata.quantization_method == "SmoothQuant W8A8"
        assert bundle.metadata.quantization_alpha == 0.5
        assert bundle.runtime_config.is_quantized is True
        assert bundle.runtime_config.quantization_format == "QDQ"
        assert bundle.runtime_config.activation_type == "QInt8"
        assert bundle.runtime_config.weight_type == "QInt8"

        # Verify both original and quantized models are referenced
        assert bundle.artifacts.model_onnx == "model.onnx"
        assert bundle.artifacts.model_original == "model_original.onnx"

    def test_create_metadata(self, mock_model_artifacts, mock_onnx_artifacts):
        """Test metadata creation from artifacts."""
        builder = BundleBuilder(alki_version="0.1.5")

        metadata = builder._create_metadata(
            mock_model_artifacts, mock_onnx_artifacts, None, "cpu", "balanced"
        )

        assert metadata.model_id == "gpt2"
        assert metadata.architecture == "GPT2LMHeadModel"
        assert metadata.alki_version == "0.1.5"
        assert metadata.target == "cpu"
        assert metadata.preset == "balanced"
        assert metadata.quantization_method is None
        assert isinstance(metadata.created_at, datetime)

    def test_create_metadata_with_quantization(
        self, mock_model_artifacts, mock_onnx_artifacts, mock_quantization_artifacts
    ):
        """Test metadata creation with quantization artifacts."""
        builder = BundleBuilder()

        metadata = builder._create_metadata(
            mock_model_artifacts,
            mock_onnx_artifacts,
            mock_quantization_artifacts,
            "cpu",
            "small",
        )

        assert metadata.quantization_method == "SmoothQuant W8A8"
        assert metadata.quantization_alpha == 0.5

    def test_create_runtime_config_cpu(self, mock_onnx_artifacts):
        """Test runtime config creation for CPU target."""
        builder = BundleBuilder()

        runtime_config = builder._create_runtime_config(
            mock_onnx_artifacts, None, "cpu"
        )

        assert runtime_config.provider == "CPUExecutionProvider"
        assert runtime_config.opset_version == 14
        assert runtime_config.use_cache is False
        assert runtime_config.is_quantized is False
        assert runtime_config.input_names == ["input_ids", "attention_mask"]
        assert runtime_config.output_names == ["logits"]

    def test_create_runtime_config_openvino(self, mock_onnx_artifacts):
        """Test runtime config creation for OpenVINO target."""
        builder = BundleBuilder()

        runtime_config = builder._create_runtime_config(
            mock_onnx_artifacts, None, "openvino"
        )

        assert runtime_config.provider == "OpenVINOExecutionProvider"

    def test_create_runtime_config_with_quantization(
        self, mock_onnx_artifacts, mock_quantization_artifacts
    ):
        """Test runtime config with quantization information."""
        builder = BundleBuilder()

        runtime_config = builder._create_runtime_config(
            mock_onnx_artifacts, mock_quantization_artifacts, "cpu"
        )

        assert runtime_config.is_quantized is True
        assert runtime_config.quantization_format == "QDQ"
        assert runtime_config.activation_type == "QInt8"
        assert runtime_config.weight_type == "QInt8"

    @patch("src.core.bundle_builder.shutil.copy2")
    def test_copy_tokenizer_artifacts_from_files(
        self, mock_copy, mock_model_artifacts, tmp_path
    ):
        """Test copying tokenizer artifacts from downloaded files."""
        builder = BundleBuilder()

        # Create mock tokenizer source directory with files
        tokenizer_source = tmp_path / "tokenizer_source"
        tokenizer_source.mkdir()

        # Create some tokenizer files
        (tokenizer_source / "tokenizer_config.json").write_text(
            '{"model_type": "gpt2"}'
        )
        (tokenizer_source / "tokenizer.json").write_text('{"version": "1.0"}')
        (tokenizer_source / "vocab.json").write_text('{"hello": 1}')

        # Update mock artifacts to point to our test directory
        mock_model_artifacts["local_path"] = tokenizer_source

        bundle_path = tmp_path / "bundle"
        bundle_path.mkdir()

        # Call the method
        builder._copy_tokenizer_artifacts(mock_model_artifacts, bundle_path)

        # Verify files were copied
        assert mock_copy.call_count >= 3  # Should copy at least 3 files

    def test_copy_tokenizer_artifacts_save_pretrained_fallback(
        self, mock_model_artifacts, tmp_path
    ):
        """Test fallback to save_pretrained when no files found."""
        builder = BundleBuilder()

        # Create an empty directory for local_path (exists but has no tokenizer files)
        empty_dir = tmp_path / "empty_model_dir"
        empty_dir.mkdir()
        mock_model_artifacts["local_path"] = str(empty_dir)

        bundle_path = tmp_path / "bundle"
        bundle_path.mkdir()

        # Call the method - should fall back to save_pretrained
        builder._copy_tokenizer_artifacts(mock_model_artifacts, bundle_path)

        # Verify save_pretrained was called
        mock_tokenizer = mock_model_artifacts["tokenizer"]
        mock_tokenizer.save_pretrained.assert_called_once_with(
            bundle_path / "tokenizer"
        )

    @patch("src.core.bundle_builder.shutil.copy2")
    def test_organize_artifacts_quantized(
        self,
        mock_copy,
        mock_model_artifacts,
        mock_onnx_artifacts,
        mock_quantization_artifacts,
        tmp_path,
    ):
        """Test artifact organization with quantized model."""
        builder = BundleBuilder()
        bundle_path = tmp_path / "bundle"
        bundle_path.mkdir()

        with patch.object(builder, "_copy_tokenizer_artifacts"):
            artifacts = builder._organize_artifacts(
                mock_model_artifacts,
                mock_onnx_artifacts,
                mock_quantization_artifacts,
                bundle_path,
            )

        # Verify artifacts structure
        assert artifacts.model_onnx == "model.onnx"
        assert artifacts.model_original == "model_original.onnx"
        assert artifacts.tokenizer_dir == "tokenizer"

        # Verify copy calls (quantized model + original model)
        assert mock_copy.call_count == 2

    def test_organize_artifacts_missing_onnx_model(
        self, mock_model_artifacts, tmp_path
    ):
        """Test error handling when ONNX model is missing."""
        builder = BundleBuilder()
        bundle_path = tmp_path / "bundle"

        # Create artifacts without valid ONNX path
        bad_onnx_artifacts = {"model_id": "gpt2"}

        with pytest.raises(FileNotFoundError, match="No valid ONNX model found"):
            builder._organize_artifacts(
                mock_model_artifacts, bad_onnx_artifacts, None, bundle_path
            )


class TestConvenienceFunction:
    """Test the convenience function for creating bundles."""

    @patch("src.core.bundle_builder.BundleBuilder.build_bundle")
    def test_create_bundle_from_pipeline(self, mock_build_bundle, tmp_path):
        """Test the convenience function calls BundleBuilder correctly."""
        mock_model_artifacts = {"model_id": "gpt2"}
        mock_onnx_artifacts = {"onnx_path": "/tmp/model.onnx"}
        mock_quantization_artifacts = None
        output_path = tmp_path / "bundle"

        # Mock return value
        mock_bundle = MagicMock()
        mock_build_bundle.return_value = mock_bundle

        result = create_bundle_from_pipeline(
            model_artifacts=mock_model_artifacts,
            onnx_artifacts=mock_onnx_artifacts,
            quantization_artifacts=mock_quantization_artifacts,
            output_path=output_path,
            target="openvino",
            preset="fast",
        )

        # Verify BundleBuilder was called correctly
        mock_build_bundle.assert_called_once_with(
            model_artifacts=mock_model_artifacts,
            onnx_artifacts=mock_onnx_artifacts,
            quantization_artifacts=mock_quantization_artifacts,
            output_path=output_path,
            target="openvino",
            preset="fast",
        )

        assert result == mock_bundle
