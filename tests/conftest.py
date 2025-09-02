"""Common test fixtures and helpers for Alki tests."""

import pytest
import numpy as np
from unittest.mock import MagicMock
from datetime import datetime
import yaml

from src.core.bundle import Bundle, BundleMetadata, RuntimeConfig, BundleArtifacts


# Test data constants
TEST_MODEL_DATA = b"fake onnx model"
TEST_TOKENIZER_CONFIG = '{"model_type": "gpt2"}'


def create_bundle_yaml_data(model_id="gpt2", quantized=False):
    """Create standard bundle YAML data structure."""
    data = {
        "metadata": {
            "model_id": model_id,
            "architecture": "GPT2LMHeadModel",
            "alki_version": "0.1.0",
            "created_at": "2024-01-15T12:00:00",
            "target": "cpu",
            "preset": "balanced",
        },
        "runtime": {"provider": "CPUExecutionProvider", "is_quantized": quantized},
        "artifacts": {
            "model_onnx": "model.onnx",
            "tokenizer_dir": "tokenizer",
            "tokenizer_config": "tokenizer/tokenizer_config.json",
        },
    }

    if quantized:
        data["metadata"]["quantization_method"] = "SmoothQuant W8A8"
        data["metadata"]["quantization_alpha"] = 0.5
        data["artifacts"]["model_original"] = "model_original.onnx"
        data["runtime"]["quantization_format"] = "QDQ"
        data["runtime"]["activation_type"] = "QInt8"
        data["runtime"]["weight_type"] = "QInt8"

    return data


def create_test_bundle_dir(tmp_path, model_id="gpt2", quantized=False):
    """Create a complete valid test bundle directory."""
    bundle_dir = tmp_path / f"test_bundle_{model_id}"
    bundle_dir.mkdir()
    (bundle_dir / "tokenizer").mkdir()

    # Create required files
    (bundle_dir / "model.onnx").write_bytes(TEST_MODEL_DATA)
    (bundle_dir / "tokenizer" / "tokenizer_config.json").write_text(
        TEST_TOKENIZER_CONFIG
    )

    if quantized:
        (bundle_dir / "model_original.onnx").write_bytes(TEST_MODEL_DATA)

    # Create bundle.yaml
    bundle_data = create_bundle_yaml_data(model_id, quantized)
    with open(bundle_dir / "bundle.yaml", "w") as f:
        yaml.safe_dump(bundle_data, f)

    return bundle_dir


def create_mock_onnx_model():
    """Create a properly mocked ONNX model with string name attributes."""
    mock_onnx_model = MagicMock()
    mock_onnx_model.graph = MagicMock()

    # Create proper mock objects for input/output with name attributes that return strings
    mock_input_ids = MagicMock()
    mock_input_ids.name = "input_ids"
    mock_attention_mask = MagicMock()
    mock_attention_mask.name = "attention_mask"
    mock_logits = MagicMock()
    mock_logits.name = "logits"

    mock_onnx_model.graph.input = [mock_input_ids, mock_attention_mask]
    mock_onnx_model.graph.output = [mock_logits]
    mock_onnx_model.graph.initializer = []

    return mock_onnx_model


def create_mock_tokenizer():
    """Create a properly mocked tokenizer that returns numpy arrays."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token = None
    mock_tokenizer.eos_token = "<|endoftext|>"
    mock_tokenizer.save_pretrained = MagicMock()
    mock_tokenizer.return_value = {
        "input_ids": np.array([[1, 2, 3, 4]], dtype=np.int64),
        "attention_mask": np.array([[1, 1, 1, 1]], dtype=np.int64),
    }
    return mock_tokenizer


@pytest.fixture
def sample_bundle_metadata():
    """Create a sample BundleMetadata for testing."""
    return BundleMetadata(
        model_id="gpt2",
        architecture="GPT2LMHeadModel",
        alki_version="0.1.0",
        created_at=datetime(2024, 1, 15, 12, 0, 0),
        target="cpu",
        preset="balanced",
        quantization_method="SmoothQuant W8A8",
        quantization_alpha=0.5,
        original_size_mb=100,
        quantized_size_mb=25,
        compression_ratio=0.25,
    )


@pytest.fixture
def sample_runtime_config():
    """Create a sample RuntimeConfig for testing."""
    return RuntimeConfig(
        provider="CPUExecutionProvider",
        is_quantized=True,
        quantization_format="QDQ",
        activation_type="QInt8",
        weight_type="QInt8",
    )


@pytest.fixture
def sample_bundle_artifacts():
    """Create sample BundleArtifacts for testing."""
    return BundleArtifacts()


@pytest.fixture
def sample_bundle(
    sample_bundle_metadata, sample_runtime_config, sample_bundle_artifacts
):
    """Create a complete sample Bundle for testing."""
    return Bundle(
        metadata=sample_bundle_metadata,
        runtime_config=sample_runtime_config,
        artifacts=sample_bundle_artifacts,
    )
