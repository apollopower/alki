"""End-to-end integration tests for Bundle pipeline."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.core.model_loader import HuggingFaceModelLoader
from src.core.onnx_exporter import OnnxExporter, OnnxExportConfig
from src.core.quantizer import (
    SmoothQuantizer,
    SmoothQuantConfig,
    CalibrationDataGenerator,
)
from src.core.bundle_builder import create_bundle_from_pipeline
from src.core.bundle_manager import load_bundle
from .conftest import create_mock_onnx_model, create_mock_tokenizer, TEST_MODEL_DATA


class TestBundleIntegration:
    """Test complete Bundle pipeline integration."""

    @pytest.fixture
    def mock_all_pipeline_dependencies(self):
        """Mock all pipeline dependencies for integration tests."""
        with (
            # HuggingFace mocks
            patch("src.core.model_loader.snapshot_download") as mock_download,
            patch("src.core.model_loader.AutoConfig.from_pretrained") as mock_config,
            patch(
                "src.core.model_loader.AutoTokenizer.from_pretrained"
            ) as mock_tokenizer,
            patch(
                "src.core.model_loader.HuggingFaceModelLoader._estimate_model_size_mb"
            ) as mock_size_est,
            # ONNX mocks
            patch("src.core.onnx_exporter.ORTModelForCausalLM") as mock_ort_model,
            patch("src.core.onnx_exporter.onnx") as mock_onnx,
            # Quantization mocks
            patch("src.core.quantizer.ort.InferenceSession") as mock_session,
            patch("src.core.quantizer.quantize_static") as mock_quantize,
        ):

            # Setup HuggingFace mocks
            mock_download.return_value = "/mock/model/path"

            config_obj = MagicMock()
            config_obj.architectures = ["GPT2LMHeadModel"]
            mock_config.return_value = config_obj

            tokenizer_obj = MagicMock()
            tokenizer_obj.pad_token = None
            tokenizer_obj.eos_token = "<|endoftext|>"

            # Use common mock tokenizer helper
            tokenizer_obj = create_mock_tokenizer()
            mock_tokenizer.return_value = tokenizer_obj

            # Mock size estimation to return a valid float
            mock_size_est.return_value = 500.0  # 500MB for test model

            # Setup ONNX mocks
            mock_model_instance = MagicMock()
            mock_ort_model.from_pretrained.return_value = mock_model_instance

            # Use common mock ONNX model helper
            mock_onnx_model = create_mock_onnx_model()
            mock_onnx.load.return_value = mock_onnx_model
            mock_onnx.checker.check_model.return_value = None

            # Setup quantization mocks
            mock_session_instance = MagicMock()
            mock_session_instance.get_inputs.return_value = [
                MagicMock(name="input_ids", shape=[1, 512], type="int64"),
                MagicMock(name="attention_mask", shape=[1, 512], type="int64"),
            ]
            mock_session_instance.run.return_value = [[0.1, 0.2, 0.3]]  # Mock logits
            mock_session.return_value = mock_session_instance

            yield {
                # HF mocks
                "download": mock_download,
                "config": mock_config,
                "tokenizer": mock_tokenizer,
                "config_obj": config_obj,
                "tokenizer_obj": tokenizer_obj,
                # ONNX mocks
                "ort_model_class": mock_ort_model,
                "ort_model_instance": mock_model_instance,
                "onnx": mock_onnx,
                "onnx_model": mock_onnx_model,
                # Quantization mocks
                "session": mock_session,
                "session_instance": mock_session_instance,
                "quantize_static": mock_quantize,
            }

    def test_complete_pipeline_without_quantization(
        self, mock_all_pipeline_dependencies
    ):
        """Test complete pipeline from model loading to bundle creation without quantization."""

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Step 1: Load model
            loader = HuggingFaceModelLoader()
            model_artifacts = loader.prepare("gpt2")

            assert model_artifacts["model_id"] == "gpt2"
            assert model_artifacts["architecture"] == "GPT2LMHeadModel"
            assert "tokenizer" in model_artifacts

            # Step 2: Export to ONNX
            onnx_output_path = temp_path / "onnx_output"
            onnx_output_path.mkdir()

            # Create test ONNX file
            fake_onnx_path = onnx_output_path / "model.onnx"
            fake_onnx_path.write_bytes(TEST_MODEL_DATA)

            exporter = OnnxExporter()

            with patch.object(exporter, "export") as mock_export:
                mock_export.return_value = {
                    "model_id": "gpt2",
                    "architecture": "GPT2LMHeadModel",
                    "onnx_path": fake_onnx_path,
                    "onnx_model": mock_all_pipeline_dependencies["onnx_model"],
                    "output_dir": onnx_output_path,
                    "export_config": OnnxExportConfig(),
                }

                onnx_artifacts = exporter.export(model_artifacts, onnx_output_path)

            # Step 3: Create bundle (no quantization)
            bundle_output_path = temp_path / "bundle_output"

            # Create necessary directories and files that the bundle builder expects
            bundle_output_path.mkdir(exist_ok=True)
            (bundle_output_path / "tokenizer").mkdir(exist_ok=True)

            # Copy the test ONNX model to the bundle directory
            (bundle_output_path / "model.onnx").write_bytes(fake_onnx_path.read_bytes())

            # Create tokenizer config file
            (bundle_output_path / "tokenizer" / "tokenizer_config.json").write_text(
                '{"model_type": "gpt2"}'
            )

            bundle = create_bundle_from_pipeline(
                model_artifacts=model_artifacts,
                onnx_artifacts=onnx_artifacts,
                quantization_artifacts=None,
                output_path=bundle_output_path,
                target="cpu",
                preset="balanced",
            )

            # Verify bundle structure
            assert bundle.metadata.model_id == "gpt2"
            assert bundle.metadata.target == "cpu"
            assert bundle.metadata.preset == "balanced"
            assert bundle.metadata.quantization_method is None
            assert bundle.runtime_config.provider == "CPUExecutionProvider"
            assert bundle.runtime_config.is_quantized is False

            # Verify bundle files
            assert bundle_output_path.exists()
            assert (bundle_output_path / "bundle.yaml").exists()
            assert (bundle_output_path / "model.onnx").exists()
            assert (bundle_output_path / "tokenizer").exists()

            # Step 4: Load bundle back
            loaded_bundle = load_bundle(bundle_output_path)
            assert loaded_bundle.metadata.model_id == "gpt2"
            assert loaded_bundle.bundle_path == bundle_output_path

    def test_complete_pipeline_with_quantization(self, mock_all_pipeline_dependencies):
        """Test complete pipeline with quantization enabled."""

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Step 1: Load model
            loader = HuggingFaceModelLoader()
            model_artifacts = loader.prepare("gpt2")

            # Step 2: Export to ONNX
            onnx_output_path = temp_path / "onnx_output"
            onnx_output_path.mkdir()

            fake_onnx_path = onnx_output_path / "model.onnx"
            fake_onnx_path.write_bytes(TEST_MODEL_DATA)

            onnx_artifacts = {
                "model_id": "gpt2",
                "architecture": "GPT2LMHeadModel",
                "onnx_path": fake_onnx_path,
                "onnx_model": mock_all_pipeline_dependencies["onnx_model"],
                "output_dir": onnx_output_path,
                "export_config": OnnxExportConfig(),
            }

            # Step 3: Quantization
            quant_config = SmoothQuantConfig(
                alpha=0.5, calibration_samples=8
            )  # Small for test
            quantizer = SmoothQuantizer(quant_config)

            # Create mock calibration data
            tokenizer = model_artifacts["tokenizer"]
            calibration_texts = ["Hello world", "Test sentence"]
            calibration_data = CalibrationDataGenerator(
                tokenizer, calibration_texts, max_length=64
            )

            quantized_model_path = temp_path / "quantized_model.onnx"

            with patch.object(quantizer, "quantize_model") as mock_quantize:
                # Create test quantized model file
                quantized_model_path.write_bytes(TEST_MODEL_DATA)
                mock_quantize.return_value = quantized_model_path

                result_path = quantizer.quantize_model(
                    fake_onnx_path, quantized_model_path, calibration_data
                )

            quantization_artifacts = {
                "quantized_model_path": result_path,
                "config": quant_config,
            }

            # Step 4: Create bundle with quantization
            bundle_output_path = temp_path / "quantized_bundle"

            # Create necessary directories and files that the bundle builder expects
            bundle_output_path.mkdir(exist_ok=True)
            (bundle_output_path / "tokenizer").mkdir(exist_ok=True)

            # Copy the test ONNX models to the bundle directory
            (bundle_output_path / "model.onnx").write_bytes(
                quantized_model_path.read_bytes()
            )
            (bundle_output_path / "model_original.onnx").write_bytes(
                fake_onnx_path.read_bytes()
            )

            # Create tokenizer config file
            (bundle_output_path / "tokenizer" / "tokenizer_config.json").write_text(
                '{"model_type": "gpt2"}'
            )

            bundle = create_bundle_from_pipeline(
                model_artifacts=model_artifacts,
                onnx_artifacts=onnx_artifacts,
                quantization_artifacts=quantization_artifacts,
                output_path=bundle_output_path,
                target="cpu",
                preset="small",
            )

            # Verify quantized bundle
            assert bundle.metadata.quantization_method == "SmoothQuant W8A8"
            assert bundle.metadata.quantization_alpha == 0.5
            assert bundle.runtime_config.is_quantized is True
            assert bundle.runtime_config.quantization_format == "QDQ"

            # Verify bundle files include both models
            assert (bundle_output_path / "model.onnx").exists()  # Quantized model
            assert (
                bundle_output_path / "model_original.onnx"
            ).exists()  # Original model

            # Step 5: Verify bundle can be loaded and validated
            loaded_bundle = load_bundle(bundle_output_path)
            assert loaded_bundle.metadata.quantization_method == "SmoothQuant W8A8"

            # Bundle validation should pass (files exist)
            issues = loaded_bundle.validate()
            assert len(issues) == 0

    def test_pipeline_error_handling_model_loading_failure(
        self, mock_all_pipeline_dependencies
    ):
        """Test pipeline error handling when model loading fails."""

        # Make model loading fail
        mock_all_pipeline_dependencies["download"].side_effect = Exception(
            "Model not found"
        )

        loader = HuggingFaceModelLoader()

        with pytest.raises(Exception, match="Model not found"):
            loader.prepare("nonexistent/model")

    def test_pipeline_error_handling_onnx_export_failure(
        self, mock_all_pipeline_dependencies
    ):
        """Test pipeline error handling when ONNX export fails."""

        # Make ONNX export fail
        mock_all_pipeline_dependencies[
            "ort_model_class"
        ].from_pretrained.side_effect = Exception("ONNX export failed")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Model loading succeeds
            loader = HuggingFaceModelLoader()
            model_artifacts = loader.prepare("gpt2")

            # ONNX export fails
            exporter = OnnxExporter()
            onnx_output_path = temp_path / "onnx_output"

            with pytest.raises(RuntimeError, match="ONNX export failed"):
                exporter.export(model_artifacts, onnx_output_path)

    def test_pipeline_with_different_targets(self, mock_all_pipeline_dependencies):
        """Test pipeline produces different configs for different targets."""

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Common setup
            loader = HuggingFaceModelLoader()
            model_artifacts = loader.prepare("gpt2")

            fake_onnx_path = temp_path / "model.onnx"
            fake_onnx_path.write_bytes(TEST_MODEL_DATA)

            onnx_artifacts = {
                "model_id": "gpt2",
                "architecture": "GPT2LMHeadModel",
                "onnx_path": fake_onnx_path,
                "onnx_model": mock_all_pipeline_dependencies["onnx_model"],
                "export_config": OnnxExportConfig(),
            }

            # Test CPU target
            cpu_bundle = create_bundle_from_pipeline(
                model_artifacts=model_artifacts,
                onnx_artifacts=onnx_artifacts,
                quantization_artifacts=None,
                output_path=temp_path / "cpu_bundle",
                target="cpu",
                preset="balanced",
            )

            assert cpu_bundle.runtime_config.provider == "CPUExecutionProvider"
            assert cpu_bundle.metadata.target == "cpu"

            # Test OpenVINO target
            openvino_bundle = create_bundle_from_pipeline(
                model_artifacts=model_artifacts,
                onnx_artifacts=onnx_artifacts,
                quantization_artifacts=None,
                output_path=temp_path / "openvino_bundle",
                target="openvino",
                preset="balanced",
            )

            assert (
                openvino_bundle.runtime_config.provider == "OpenVINOExecutionProvider"
            )
            assert openvino_bundle.metadata.target == "openvino"

    def test_roundtrip_bundle_serialization(self, mock_all_pipeline_dependencies):
        """Test that bundle can be serialized and deserialized correctly."""

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create bundle
            loader = HuggingFaceModelLoader()
            model_artifacts = loader.prepare("gpt2")

            fake_onnx_path = temp_path / "model.onnx"
            fake_onnx_path.write_bytes(TEST_MODEL_DATA)

            onnx_artifacts = {
                "model_id": "gpt2",
                "architecture": "GPT2LMHeadModel",
                "onnx_path": fake_onnx_path,
                "onnx_model": mock_all_pipeline_dependencies["onnx_model"],
                "export_config": OnnxExportConfig(),
            }

            bundle_path = temp_path / "roundtrip_bundle"

            # Create necessary directories and files that the bundle builder expects
            bundle_path.mkdir(exist_ok=True)
            (bundle_path / "tokenizer").mkdir(exist_ok=True)

            # Copy the test ONNX model to the bundle directory
            (bundle_path / "model.onnx").write_bytes(fake_onnx_path.read_bytes())

            # Create tokenizer config file
            (bundle_path / "tokenizer" / "tokenizer_config.json").write_text(
                '{"model_type": "gpt2"}'
            )

            original_bundle = create_bundle_from_pipeline(
                model_artifacts=model_artifacts,
                onnx_artifacts=onnx_artifacts,
                quantization_artifacts=None,
                output_path=bundle_path,
                target="cpu",
                preset="balanced",
            )

            # Load bundle back
            loaded_bundle = load_bundle(bundle_path)

            # Compare key attributes
            assert loaded_bundle.metadata.model_id == original_bundle.metadata.model_id
            assert loaded_bundle.metadata.target == original_bundle.metadata.target
            assert loaded_bundle.metadata.preset == original_bundle.metadata.preset
            assert (
                loaded_bundle.runtime_config.provider
                == original_bundle.runtime_config.provider
            )
            assert (
                loaded_bundle.runtime_config.is_quantized
                == original_bundle.runtime_config.is_quantized
            )
            assert (
                loaded_bundle.artifacts.model_onnx
                == original_bundle.artifacts.model_onnx
            )

            # Verify bundle path is set correctly
            assert loaded_bundle.bundle_path == bundle_path
