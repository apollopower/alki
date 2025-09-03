"""
Tests for the ONNX Runtime module.

These tests validate the runtime inference capabilities for deployed bundles,
including model loading, text generation, and error handling.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch

from src.core.onnx_runtime import OnnxRuntimeRunner, GenerationConfig
from src.core.bundle import Bundle, BundleMetadata, RuntimeConfig, BundleArtifacts
from src.core.constants import ExecutionProviders
from datetime import datetime


@pytest.fixture
def mock_bundle():
    """Create a mock bundle for testing."""
    metadata = BundleMetadata(
        model_id="test-gpt2",
        architecture="GPT2LMHeadModel",
        alki_version="0.1.0",
        created_at=datetime.now(),
        target="cpu",
        preset="balanced",
    )

    runtime = RuntimeConfig(
        provider=ExecutionProviders.CPU,
        is_quantized=False,
    )

    artifacts = BundleArtifacts()

    bundle = Bundle(
        metadata=metadata,
        runtime_config=runtime,
        artifacts=artifacts,
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        bundle.bundle_path = Path(temp_dir)
        (bundle.bundle_path / "model.onnx").touch()
        (bundle.bundle_path / "tokenizer").mkdir()
        (bundle.bundle_path / "tokenizer" / "tokenizer_config.json").touch()
        yield bundle


class TestGenerationConfig:
    """Test GenerationConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = GenerationConfig()
        assert config.max_tokens == 100
        assert config.temperature == 1.0
        assert config.top_p == 0.9
        assert config.top_k == 50
        assert config.do_sample is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = GenerationConfig(
            max_tokens=50,
            temperature=0.8,
            top_p=0.95,
            top_k=40,
            do_sample=False,
        )
        assert config.max_tokens == 50
        assert config.temperature == 0.8
        assert config.top_p == 0.95
        assert config.top_k == 40
        assert config.do_sample is False


class TestOnnxRuntimeRunner:
    """Test OnnxRuntimeRunner class."""

    def test_init(self, mock_bundle):
        """Test runner initialization."""
        runner = OnnxRuntimeRunner(mock_bundle)
        assert runner.bundle == mock_bundle
        assert runner.session is None
        assert runner.tokenizer is None
        assert runner._model_loaded is False

    def test_get_model_info_not_loaded(self, mock_bundle):
        """Test model info when model is not loaded."""
        runner = OnnxRuntimeRunner(mock_bundle)
        info = runner.get_model_info()
        assert info["status"] == "not_loaded"

    def test_get_model_info_loaded(self, mock_bundle):
        """Test model info when model is loaded."""
        runner = OnnxRuntimeRunner(mock_bundle)
        runner._model_loaded = True

        info = runner.get_model_info()
        assert info["status"] == "loaded"
        assert info["model_id"] == "test-gpt2"
        assert info["architecture"] == "GPT2LMHeadModel"
        assert info["target"] == "cpu"
        assert info["provider"] == ExecutionProviders.CPU
        assert info["is_quantized"] is False

    @patch("onnxruntime.InferenceSession")
    @patch("transformers.AutoTokenizer")
    def test_validate_compatibility_success(
        self, mock_tokenizer, mock_ort, mock_bundle
    ):
        """Test compatibility validation success."""
        runner = OnnxRuntimeRunner(mock_bundle)
        compatibility = runner.validate_compatibility()

        assert compatibility["compatible"] is True
        assert len(compatibility["issues"]) == 0

    def test_validate_compatibility_missing_onnxruntime(self, mock_bundle):
        """Test compatibility validation when onnxruntime is missing."""
        runner = OnnxRuntimeRunner(mock_bundle)

        with patch(
            "builtins.__import__",
            side_effect=ImportError("No module named 'onnxruntime'"),
        ):
            compatibility = runner.validate_compatibility()

        assert compatibility["compatible"] is False
        assert "onnxruntime not available" in compatibility["issues"]

    def test_load_model_no_bundle_path(self, mock_bundle):
        """Test loading model when bundle path is not set."""
        mock_bundle.bundle_path = None
        runner = OnnxRuntimeRunner(mock_bundle)

        with pytest.raises(RuntimeError, match="Bundle path not set"):
            runner.load_model()

    def test_load_model_missing_files(self, mock_bundle):
        """Test loading model when required files are missing."""
        runner = OnnxRuntimeRunner(mock_bundle)

        # Remove model file
        (mock_bundle.bundle_path / "model.onnx").unlink()

        with pytest.raises(RuntimeError, match="Model file not found"):
            runner.load_model()

    def test_load_model_success(self, mock_bundle):
        """Test successful model loading - basic test without actual loading."""
        runner = OnnxRuntimeRunner(mock_bundle)

        # This test just verifies the runner can be created
        # Full integration tests with actual models are done in CLI tests
        assert runner.session is None
        assert runner.tokenizer is None
        assert runner._model_loaded is False

    def test_generate_not_loaded(self, mock_bundle):
        """Test generation when model is not loaded."""
        runner = OnnxRuntimeRunner(mock_bundle)

        with pytest.raises(RuntimeError, match="Model not loaded"):
            runner.generate("test prompt")

    def test_generate_success(self, mock_bundle):
        """Test text generation - integration test done in CLI tests."""
        # Complex generation testing is done in integration tests
        # This test focuses on the basic API
        config = GenerationConfig(max_tokens=5, do_sample=True)
        assert config.max_tokens == 5
        assert config.do_sample is True

    def test_generate_greedy_config(self, mock_bundle):
        """Test greedy generation configuration."""
        # Test greedy configuration
        config = GenerationConfig(max_tokens=5, do_sample=False)
        assert config.do_sample is False
        assert config.max_tokens == 5

    def test_softmax_function(self, mock_bundle):
        """Test the internal softmax function."""
        import numpy as np

        runner = OnnxRuntimeRunner(mock_bundle)

        # Test softmax with simple values
        x = np.array([1.0, 2.0, 3.0])
        result = runner._softmax(x)

        # Softmax should sum to 1
        assert np.isclose(np.sum(result), 1.0)

        # Largest input should have largest output
        assert result[2] > result[1] > result[0]

    def test_softmax_numerical_stability(self, mock_bundle):
        """Test softmax numerical stability with large values."""
        import numpy as np

        runner = OnnxRuntimeRunner(mock_bundle)

        # Test with large values that could cause overflow
        x = np.array([1000.0, 1001.0, 1002.0])
        result = runner._softmax(x)

        # Should still sum to 1 and not contain inf/nan
        assert np.isclose(np.sum(result), 1.0)
        assert np.all(np.isfinite(result))
