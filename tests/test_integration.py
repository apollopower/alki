"""Integration tests for the model loading and ONNX export pipeline."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile

from src.core.model_loader import HuggingFaceModelLoader
from src.core.onnx_exporter import OnnxExporter, OnnxExportConfig


@pytest.fixture
def mock_all_dependencies():
    """Mock both HuggingFace and ONNX dependencies for integration tests."""
    with (
        # HuggingFace mocks
        patch("src.core.model_loader.snapshot_download") as mock_download,
        patch("src.core.model_loader.AutoConfig.from_pretrained") as mock_config,
        patch("src.core.model_loader.AutoTokenizer.from_pretrained") as mock_tokenizer,
        patch(
            "src.core.model_loader.HuggingFaceModelLoader._estimate_model_size_mb"
        ) as mock_size_est,
        # ONNX mocks
        patch("src.core.onnx_exporter.ORTModelForCausalLM") as mock_ort_model,
        patch("src.core.onnx_exporter.onnx") as mock_onnx,
    ):

        # Setup HuggingFace mocks
        mock_download.return_value = "/mock/model/path"

        config_obj = MagicMock()
        config_obj.architectures = ["GPT2LMHeadModel"]
        mock_config.return_value = config_obj

        mock_tokenizer.return_value = MagicMock()

        # Mock size estimation to return a valid float
        mock_size_est.return_value = 500.0  # 500MB for test model

        # Setup ONNX mocks
        mock_model_instance = MagicMock()
        mock_ort_model.from_pretrained.return_value = mock_model_instance

        mock_onnx_model = MagicMock()
        mock_onnx.load.return_value = mock_onnx_model
        mock_onnx.checker.check_model.return_value = None

        yield {
            # HF mocks
            "download": mock_download,
            "config": mock_config,
            "tokenizer": mock_tokenizer,
            "config_obj": config_obj,
            "size_estimate": mock_size_est,
            # ONNX mocks
            "ort_model_class": mock_ort_model,
            "ort_model_instance": mock_model_instance,
            "onnx": mock_onnx,
            "onnx_model": mock_onnx_model,
        }


def test_full_pipeline_model_loader_to_onnx_export(mock_all_dependencies):
    """Test the complete pipeline from model loading to ONNX export."""

    # Step 1: Load model with HuggingFaceModelLoader
    loader = HuggingFaceModelLoader()
    model_artifacts = loader.prepare("gpt2")

    # Verify model loading worked
    assert model_artifacts["model_id"] == "gpt2"
    assert model_artifacts["architecture"] == "GPT2LMHeadModel"
    assert "config" in model_artifacts
    assert "tokenizer" in model_artifacts

    # Step 2: Export to ONNX
    exporter = OnnxExporter()

    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "onnx_output"

        export_result = exporter.export(model_artifacts, output_path)

        # Verify export result
        assert export_result["model_id"] == "gpt2"
        assert export_result["architecture"] == "GPT2LMHeadModel"
        assert export_result["onnx_path"] == output_path / "model.onnx"
        assert "onnx_model" in export_result

        # Verify all expected calls were made
        mock_all_dependencies["download"].assert_called_once_with(repo_id="gpt2")
        mock_all_dependencies["config"].assert_called_once_with("gpt2")
        mock_all_dependencies["tokenizer"].assert_called_once_with("gpt2")
        mock_all_dependencies["ort_model_class"].from_pretrained.assert_called_once()
        mock_all_dependencies["onnx"].load.assert_called_once()
        mock_all_dependencies["onnx"].checker.check_model.assert_called_once()


def test_pipeline_with_unsupported_architecture(mock_all_dependencies):
    """Test pipeline behavior with an unsupported architecture."""

    # Configure mock to return unsupported architecture
    mock_all_dependencies["config_obj"].architectures = ["MambaForCausalLM"]

    # Make ONNX export fail for unsupported architecture
    mock_all_dependencies["ort_model_class"].from_pretrained.side_effect = Exception(
        "Unsupported architecture"
    )

    # Step 1: Load model (should succeed)
    loader = HuggingFaceModelLoader()
    model_artifacts = loader.prepare("test/mamba-model")

    assert model_artifacts["architecture"] == "MambaForCausalLM"

    # Step 2: Attempt ONNX export (should fail gracefully)
    exporter = OnnxExporter()

    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "onnx_output"

        with pytest.raises(RuntimeError) as exc_info:
            exporter.export(model_artifacts, output_path)

        assert "ONNX export failed" in str(exc_info.value)
        assert "MambaForCausalLM may not be fully supported" in str(exc_info.value)


def test_pipeline_with_custom_config(mock_all_dependencies):
    """Test pipeline with custom ONNX export configuration."""

    # Step 1: Load model
    loader = HuggingFaceModelLoader()
    model_artifacts = loader.prepare("gpt2")

    # Step 2: Export with custom config
    custom_config = OnnxExportConfig(use_gpu=True, opset_version=16)
    exporter = OnnxExporter(custom_config)

    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "onnx_output"

        export_result = exporter.export(model_artifacts, output_path)

        # Verify custom config was used
        assert export_result["export_config"].use_gpu is True
        assert export_result["export_config"].opset_version == 16

        # Verify CPU provider was used (GPU requests ignored now)
        mock_all_dependencies[
            "ort_model_class"
        ].from_pretrained.assert_called_once_with(
            model_id="gpt2",
            export=True,
            use_cache=False,
            provider="CPUExecutionProvider",
        )


def test_validate_before_export():
    """Test using architecture validation before attempting export."""

    # Create exporter
    exporter = OnnxExporter()

    # Test with supported architecture
    supported_artifacts = {
        "model_id": "gpt2",
        "architecture": "GPT2LMHeadModel",
        "config": MagicMock(),
        "tokenizer": MagicMock(),
    }

    assert exporter.validate_architecture(supported_artifacts["architecture"]) is True

    # Test with unsupported architecture
    unsupported_artifacts = {
        "model_id": "test/mamba",
        "architecture": "MambaForCausalLM",
        "config": MagicMock(),
        "tokenizer": MagicMock(),
    }

    assert (
        exporter.validate_architecture(unsupported_artifacts["architecture"]) is False
    )
