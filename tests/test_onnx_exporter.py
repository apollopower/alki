"""Tests for the ONNX exporter."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile

from src.core.onnx_exporter import OnnxExporter, OnnxExportConfig


@pytest.fixture
def mock_model_artifacts():
    """Sample model artifacts from HuggingFaceModelLoader."""
    return {
        "model_id": "gpt2",
        "local_path": Path("/mock/model/path"),
        "config": MagicMock(),
        "tokenizer": MagicMock(),
        "architecture": "GPT2LMHeadModel",
    }


@pytest.fixture
def mock_onnx_dependencies():
    """Mock all ONNX-related dependencies."""
    with (
        patch("src.core.onnx_exporter.ORTModelForCausalLM") as mock_ort_model,
        patch("src.core.onnx_exporter.onnx") as mock_onnx,
    ):
        # Setup ORTModel mock
        mock_model_instance = MagicMock()
        mock_ort_model.from_pretrained.return_value = mock_model_instance

        # Setup ONNX validation mock
        mock_onnx_model = MagicMock()
        mock_onnx.load.return_value = mock_onnx_model
        mock_onnx.checker.check_model.return_value = None

        yield {
            "ort_model_class": mock_ort_model,
            "ort_model_instance": mock_model_instance,
            "onnx": mock_onnx,
            "onnx_model": mock_onnx_model,
        }


def test_onnx_export_config_defaults():
    """Test default configuration values."""
    config = OnnxExportConfig()

    assert config.opset_version == 14
    assert config.use_gpu is False
    assert config.optimize is True
    assert config.output_dir is None
    assert config.dynamic_axes is not None
    assert "input_ids" in config.dynamic_axes
    assert "attention_mask" in config.dynamic_axes


def test_onnx_export_config_custom_values():
    """Test custom configuration values."""
    config = OnnxExportConfig(
        opset_version=16, use_gpu=True, optimize=False, output_dir=Path("/custom/path")
    )

    assert config.opset_version == 16
    assert config.use_gpu is True
    assert config.optimize is False
    assert config.output_dir == Path("/custom/path")


def test_export_successful_cpu(mock_onnx_dependencies, mock_model_artifacts):
    """Test successful ONNX export with CPU provider."""
    exporter = OnnxExporter()

    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "onnx_output"

        result = exporter.export(mock_model_artifacts, output_path)

        # Verify the result structure
        assert result["model_id"] == "gpt2"
        assert result["architecture"] == "GPT2LMHeadModel"
        assert result["onnx_path"] == output_path / "model.onnx"
        assert result["output_dir"] == output_path
        assert "onnx_model" in result
        assert "export_config" in result

        # Verify method calls (parameters are now passed as keyword arguments)
        mock_onnx_dependencies[
            "ort_model_class"
        ].from_pretrained.assert_called_once_with(
            model_id="gpt2",
            export=True,
            use_cache=False,
            provider="CPUExecutionProvider",
        )

        # Verify output directory was created
        assert output_path.exists()

        # Verify save was called
        mock_onnx_dependencies[
            "ort_model_instance"
        ].save_pretrained.assert_called_once_with(output_path)

        # Verify ONNX validation
        mock_onnx_dependencies["onnx"].load.assert_called_once()
        mock_onnx_dependencies["onnx"].checker.check_model.assert_called_once()


def test_export_cpu_only_config(mock_onnx_dependencies, mock_model_artifacts):
    """Test ONNX export - only CPU supported now."""
    config = OnnxExportConfig(use_gpu=True)  # GPU config ignored
    exporter = OnnxExporter(config)

    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "onnx_output"

        exporter.export(mock_model_artifacts, output_path)

        # Verify CPU provider was used (GPU option ignored)
        mock_onnx_dependencies[
            "ort_model_class"
        ].from_pretrained.assert_called_once_with(
            model_id="gpt2",
            export=True,
            use_cache=False,
            provider="CPUExecutionProvider",
        )


def test_export_failure_handling(mock_onnx_dependencies, mock_model_artifacts):
    """Test error handling during export."""
    # Make ORTModel.from_pretrained raise an exception
    mock_onnx_dependencies["ort_model_class"].from_pretrained.side_effect = Exception(
        "Export failed"
    )

    exporter = OnnxExporter()

    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "onnx_output"

        with pytest.raises(RuntimeError) as exc_info:
            exporter.export(mock_model_artifacts, output_path)

        assert "ONNX export failed for gpt2" in str(exc_info.value)
        assert "Export failed" in str(exc_info.value)


def test_export_failure_unsupported_architecture(mock_onnx_dependencies):
    """Test error handling for unsupported architecture."""
    # Mock artifacts with unsupported architecture
    unsupported_artifacts = {
        "model_id": "test/mamba-model",
        "local_path": Path("/mock/path"),
        "config": MagicMock(),
        "tokenizer": MagicMock(),
        "architecture": "MambaForCausalLM",
    }

    # Make export fail
    mock_onnx_dependencies["ort_model_class"].from_pretrained.side_effect = Exception(
        "Unsupported"
    )

    exporter = OnnxExporter()

    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "onnx_output"

        with pytest.raises(RuntimeError) as exc_info:
            exporter.export(unsupported_artifacts, output_path)

        error_message = str(exc_info.value)
        assert "ONNX export failed" in error_message
        assert "MambaForCausalLM may not be fully supported" in error_message


def test_validate_architecture_supported():
    """Test architecture validation for supported models."""
    exporter = OnnxExporter()

    # Test known supported architectures
    assert exporter.validate_architecture("GPT2LMHeadModel") is True
    assert exporter.validate_architecture("BertForSequenceClassification") is True
    assert exporter.validate_architecture("LlamaForCausalLM") is True
    assert exporter.validate_architecture("MistralForCausalLM") is True


def test_validate_architecture_unsupported():
    """Test architecture validation for unsupported models."""
    exporter = OnnxExporter()

    # Test known unsupported architectures
    assert exporter.validate_architecture("MambaForCausalLM") is False
    assert exporter.validate_architecture("MixtralForCausalLM") is False
    assert exporter.validate_architecture("Phi3ForCausalLM") is False


def test_validate_architecture_unknown():
    """Test architecture validation for unknown models."""
    exporter = OnnxExporter()

    # Unknown architecture should default to unsupported (False)
    assert exporter.validate_architecture("SomeNewModelArchitecture") is False


def test_exporter_with_custom_config(mock_onnx_dependencies, mock_model_artifacts):
    """Test exporter with custom configuration."""
    custom_config = OnnxExportConfig(opset_version=16, use_gpu=True, optimize=False)

    exporter = OnnxExporter(custom_config)

    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "onnx_output"

        result = exporter.export(mock_model_artifacts, output_path)

        # Verify custom config is preserved in result
        assert result["export_config"] == custom_config
        assert result["export_config"].opset_version == 16
        assert result["export_config"].use_gpu is True


def test_export_with_low_memory_mode(mock_onnx_dependencies, mock_model_artifacts):
    """Test ONNX export with low memory mode enabled."""
    # Configure for low memory mode
    config = OnnxExportConfig(low_memory=True, memory_threshold_mb=2000.0)

    with patch("src.core.onnx_exporter.MemoryManager") as mock_memory_manager_class:
        # Setup mock memory manager
        mock_memory_manager = mock_memory_manager_class.return_value
        mock_memory_manager.get_memory_info.return_value = {
            "available_mb": 4000.0,
            "used_pct": 60.0,
            "total_mb": 8000.0,
        }
        mock_memory_manager.check_memory_threshold.return_value = (True, "Memory OK")

        exporter = OnnxExporter(config)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "onnx_output"

            # Add size_mb to mock artifacts
            mock_model_artifacts["size_mb"] = 1000.0

            result = exporter.export(mock_model_artifacts, output_path)

            # Verify low memory mode was activated
            mock_memory_manager.set_low_memory_mode.assert_called_once()
            # Garbage collection should be called at least once (pre-export + context manager)
            assert mock_memory_manager.force_garbage_collection.call_count >= 1

            # Verify export still succeeded
            assert result["model_id"] == "gpt2"


def test_export_fails_gracefully_when_memory_insufficient(
    mock_onnx_dependencies, mock_model_artifacts
):
    """Test graceful failure when insufficient memory is detected."""
    config = OnnxExportConfig(low_memory=True)

    with patch("src.core.onnx_exporter.MemoryManager") as mock_memory_manager_class:
        # Setup mock memory manager with insufficient memory
        mock_memory_manager = mock_memory_manager_class.return_value
        mock_memory_manager.get_memory_info.return_value = {
            "available_mb": 2000.0,  # Only 2GB available
            "used_pct": 75.0,
            "total_mb": 8000.0,
        }

        exporter = OnnxExporter(config)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "onnx_output"

            # Mock large model requiring more memory than available
            mock_model_artifacts["size_mb"] = (
                2200.0  # TinyLlama size (~6.6GB requirement)
            )

            with pytest.raises(RuntimeError) as exc_info:
                exporter.export(mock_model_artifacts, output_path)

            error_message = str(exc_info.value)
            assert "Model too large for available memory" in error_message
            assert "Estimated requirement: 6600.0MB" in error_message
            assert "Available memory: 2000.0MB" in error_message
            assert "Try a smaller model" in error_message
