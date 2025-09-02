"""
Unit tests for SmoothQuant quantization module.

These tests use mocked models and data to verify the quantization logic
without requiring actual model downloads or long processing times.
All tests should complete in under 1 second.
"""

import pytest
import numpy as np
import onnx
from onnx import helper, TensorProto
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile

from src.core.quantizer import (
    SmoothQuantConfig,
    SmoothQuantizer,
    CalibrationDataGenerator,
    create_default_calibration_texts,
)


class TestSmoothQuantConfig:
    """Test configuration validation and defaults."""

    def test_default_config(self):
        """Verify default configuration values."""
        config = SmoothQuantConfig()
        assert config.alpha == 0.5
        assert config.calibration_samples == 128
        assert config.per_channel is True
        assert config.symmetric is True
        assert config.opset_version == 14

    def test_alpha_validation(self):
        """Alpha must be between 0 and 1."""
        # Valid alphas
        SmoothQuantConfig(alpha=0.0)
        SmoothQuantConfig(alpha=0.5)
        SmoothQuantConfig(alpha=1.0)

        # Invalid alphas
        with pytest.raises(ValueError, match="Alpha must be between 0 and 1"):
            SmoothQuantConfig(alpha=-0.1)

        with pytest.raises(ValueError, match="Alpha must be between 0 and 1"):
            SmoothQuantConfig(alpha=1.1)

    def test_calibration_samples_validation(self):
        """Calibration samples must be positive."""
        SmoothQuantConfig(calibration_samples=1)
        SmoothQuantConfig(calibration_samples=1000)

        with pytest.raises(ValueError, match="Need at least 1 calibration sample"):
            SmoothQuantConfig(calibration_samples=0)

        with pytest.raises(ValueError, match="Need at least 1 calibration sample"):
            SmoothQuantConfig(calibration_samples=-1)


class TestCalibrationDataGenerator:
    """Test calibration data generation."""

    def test_data_generation(self):
        """Test that calibration data is generated correctly."""
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            "input_ids": np.array([[1, 2, 3, 4, 5]]),
            "attention_mask": np.array([[1, 1, 1, 1, 1]]),
        }

        texts = ["Hello world", "Test text"]
        generator = CalibrationDataGenerator(mock_tokenizer, texts, max_length=5)

        # First sample
        sample1 = generator.get_next()
        assert sample1 is not None
        assert "input_ids" in sample1
        assert "attention_mask" in sample1

        # Second sample
        sample2 = generator.get_next()
        assert sample2 is not None

        # Should be exhausted
        sample3 = generator.get_next()
        assert sample3 is None

        # Test rewind
        generator.rewind()
        sample4 = generator.get_next()
        assert sample4 is not None


class TestSmoothQuantizer:
    """Test the main quantization logic."""

    def create_mock_onnx_model(self):
        """Create a simple mock ONNX model for testing."""
        # Create a minimal ONNX graph with one MatMul operation
        # Input (1, 768) x Weight (768, 768) = Output (1, 768)

        input_tensor = helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, [1, 768]
        )
        output_tensor = helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [1, 768]
        )

        # Create weight initializer
        weight_data = np.random.randn(768, 768).astype(np.float32)
        weight_initializer = helper.make_tensor(
            "weight", TensorProto.FLOAT, [768, 768], weight_data.flatten().tolist()
        )

        # Create MatMul node
        matmul_node = helper.make_node(
            "MatMul", inputs=["input", "weight"], outputs=["output"], name="matmul"
        )

        # Create graph
        graph = helper.make_graph(
            [matmul_node],
            "test_model",
            [input_tensor],
            [output_tensor],
            [weight_initializer],
        )

        # Create model with appropriate IR version
        model = helper.make_model(graph)
        model.ir_version = 8  # Use IR version 8 for compatibility
        model.opset_import[0].version = 14
        return model

    def test_quantizer_initialization(self):
        """Test quantizer initialization with various configs."""
        # Default config
        quantizer = SmoothQuantizer()
        assert quantizer.config.alpha == 0.5

        # Custom config
        custom_config = SmoothQuantConfig(alpha=0.7, calibration_samples=256)
        quantizer = SmoothQuantizer(custom_config)
        assert quantizer.config.alpha == 0.7
        assert quantizer.config.calibration_samples == 256

    def test_calculate_smoothing_scales(self):
        """Test the mathematical calculation of smoothing scales."""
        config = SmoothQuantConfig(alpha=0.5)
        quantizer = SmoothQuantizer(config)

        # Mock activation and weight scales
        # Activation has outlier (100), weight is normal (1)
        activation_scales = {"layer1": np.array([100.0, 1.0, 1.0])}
        weight_scales = {"layer1": np.array([1.0, 1.0, 1.0])}

        smoothing_scales = quantizer.calculate_smoothing_scales(
            activation_scales, weight_scales
        )

        # With alpha=0.5, scale = (act/weight)^(alpha/2) = (100/1)^(0.5/2) = 100^0.25 ≈ 3.16
        assert "layer1" in smoothing_scales
        scales = smoothing_scales["layer1"]
        assert np.isclose(scales[0], np.power(100.0, 0.25), rtol=0.01)
        assert np.isclose(scales[1], 1.0, rtol=0.01)
        assert np.isclose(scales[2], 1.0, rtol=0.01)

    def test_smoothing_scales_edge_cases(self):
        """Test smoothing calculation with edge cases."""
        config = SmoothQuantConfig(alpha=0.5)
        quantizer = SmoothQuantizer(config)

        # Test with alpha=0 (no smoothing)
        quantizer.config.alpha = 0
        activation_scales = {"layer1": np.array([100.0])}
        weight_scales = {"layer1": np.array([1.0])}

        smoothing_scales = quantizer.calculate_smoothing_scales(
            activation_scales, weight_scales
        )
        # With alpha=0, scale should be 1 (no smoothing)
        assert np.isclose(smoothing_scales["layer1"][0], 1.0, rtol=0.01)

        # Test with alpha=1 (maximum smoothing)
        quantizer.config.alpha = 1
        smoothing_scales = quantizer.calculate_smoothing_scales(
            activation_scales, weight_scales
        )
        # With alpha=1, scale = (act/weight)^(1/2) = sqrt(100/1) = 10
        assert np.isclose(smoothing_scales["layer1"][0], 10.0, rtol=0.01)

    @patch("onnxruntime.InferenceSession")
    def test_collect_activation_statistics(self, mock_session):
        """Test activation statistics collection."""
        # Setup
        config = SmoothQuantConfig(calibration_samples=2)
        quantizer = SmoothQuantizer(config)

        # Mock ONNX Runtime session
        mock_instance = Mock()
        mock_instance.run.return_value = [np.array([[1.0, 2.0, 3.0]])]
        mock_instance.get_inputs.return_value = []  # Empty inputs list for test
        mock_session.return_value = mock_instance

        # Mock calibration data
        mock_calibration = Mock()
        mock_calibration.get_next.side_effect = [
            {"input_ids": np.array([[1, 2, 3]])},
            {"input_ids": np.array([[4, 5, 6]])},
            None,
        ]
        mock_calibration.rewind = Mock()

        # Test
        with tempfile.NamedTemporaryFile(suffix=".onnx") as tmp:
            stats = quantizer.collect_activation_statistics(
                Path(tmp.name), mock_calibration
            )

        # Verify
        assert "input_ids" in stats
        assert len(stats["input_ids"]) == 3  # 3 values per input

    def test_quantize_model_integration(self):
        """Integration test with actual quantization (but mocked calibration)."""
        # This test actually runs quantize_static with a simple model
        # It's slower than unit tests but faster than full integration
        config = SmoothQuantConfig(alpha=0, calibration_samples=1)
        quantizer = SmoothQuantizer(config)

        # Create a simple model
        mock_model = self.create_mock_onnx_model()

        # Mock calibration data
        class SimpleCalibration:
            def __init__(self):
                self.count = 0

            def get_next(self):
                if self.count < 1:
                    self.count += 1
                    return {"input": np.random.randn(1, 768).astype(np.float32)}
                return None

            def rewind(self):
                self.count = 0

        calibration = SimpleCalibration()

        # Test with real files
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.onnx"
            output_path = Path(tmpdir) / "output.onnx"

            # Save model
            onnx.save(mock_model, str(input_path))

            # Mock only the statistics collection to speed up test
            with patch.object(
                quantizer, "collect_activation_statistics"
            ) as mock_collect:
                mock_collect.return_value = {}

                # Run quantization
                result_path = quantizer.quantize_model(
                    input_path, output_path, calibration
                )

            # Verify output exists and is valid
            assert result_path == output_path
            assert output_path.exists()

            # Verify it's a valid ONNX model
            quantized = onnx.load(str(output_path))
            assert quantized is not None


class TestDefaultCalibrationTexts:
    """Test default calibration text generation."""

    def test_creates_diverse_texts(self):
        """Verify that default texts are diverse and reasonable."""
        texts = create_default_calibration_texts()

        assert len(texts) >= 8  # Should have multiple samples
        assert all(isinstance(t, str) for t in texts)
        assert all(len(t) > 10 for t in texts)  # Non-trivial texts

        # Check for diversity (no duplicates)
        assert len(texts) == len(set(texts))
