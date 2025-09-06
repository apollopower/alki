"""Unit tests for CLI functionality."""

import pytest
from unittest.mock import patch, MagicMock
from typer.testing import CliRunner

from src.cli.main import app
from .conftest import create_mock_onnx_model, create_mock_tokenizer


class TestCLI:
    """Test CLI command functionality."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    def test_cli_help(self, runner):
        """Test CLI help output."""
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "Alki" in result.stdout
        assert "build" in result.stdout
        assert "info" in result.stdout
        assert "list" in result.stdout

    @patch("src.cli.main.HuggingFaceModelLoader")
    @patch("src.cli.main.OnnxExporter")
    @patch("src.cli.main.SmoothQuantizer")
    @patch("src.cli.main.create_bundle_from_pipeline")
    def test_build_command_success(
        self,
        mock_create_bundle,
        mock_quantizer_class,
        mock_exporter_class,
        mock_loader_class,
        runner,
        tmp_path,
    ):
        """Test successful build command."""
        # Mock the pipeline components
        mock_loader = MagicMock()
        mock_loader_class.return_value = mock_loader
        # Use common mock tokenizer helper
        mock_tokenizer = create_mock_tokenizer()

        mock_loader.prepare.return_value = {
            "model_id": "gpt2",
            "architecture": "GPT2LMHeadModel",
            "tokenizer": mock_tokenizer,
        }

        mock_exporter = MagicMock()
        mock_exporter_class.return_value = mock_exporter
        # Create proper export config mock
        export_config_mock = MagicMock()
        export_config_mock.opset_version = 14
        export_config_mock.use_cache = False
        export_config_mock.use_gpu = False

        # Use common mock ONNX model helper
        mock_onnx_model = create_mock_onnx_model()

        mock_exporter.export.return_value = {
            "onnx_path": tmp_path / "model.onnx",
            "export_config": export_config_mock,
            "onnx_model": mock_onnx_model,
        }

        mock_quantizer = MagicMock()
        mock_quantizer_class.return_value = mock_quantizer
        mock_quantizer.quantize_model.return_value = tmp_path / "quantized.onnx"

        # Mock bundle creation
        mock_bundle = MagicMock()
        mock_bundle.metadata = MagicMock()
        mock_bundle.metadata.model_id = "gpt2"
        mock_bundle.metadata.target = "cpu"
        mock_bundle.metadata.original_size_mb = 100
        mock_bundle.metadata.quantized_size_mb = 25
        mock_bundle.metadata.compression_ratio = 0.25
        mock_bundle.metadata.quantization_method = "SmoothQuant W8A8"
        mock_bundle.metadata.created_at = MagicMock()
        mock_bundle.metadata.created_at.strftime.return_value = "2024-01-15 12:00:00"
        mock_bundle.metadata.alki_version = "0.1.0"
        mock_bundle.bundle_path = tmp_path / "gpt2-cpu"
        mock_bundle.validate.return_value = []
        mock_create_bundle.return_value = mock_bundle

        # Run build command
        result = runner.invoke(
            app,
            [
                "build",
                "gpt2",
                "--output",
                str(tmp_path),
                "--target",
                "cpu",
                "--preset",
                "balanced",
            ],
        )

        assert result.exit_code == 0
        assert "Building bundle for gpt2" in result.stdout
        assert "✓" in result.stdout  # Success indicators

        # Verify pipeline was called
        mock_loader.prepare.assert_called_once_with("gpt2")
        mock_exporter.export.assert_called_once()
        mock_create_bundle.assert_called_once()

    @patch("src.cli.main.HuggingFaceModelLoader")
    def test_build_command_failure(self, mock_loader_class, runner, tmp_path):
        """Test build command failure handling."""
        # Make model loading fail
        mock_loader = MagicMock()
        mock_loader_class.return_value = mock_loader
        mock_loader.prepare.side_effect = Exception("Model not found")

        result = runner.invoke(
            app, ["build", "nonexistent/model", "--output", str(tmp_path)]
        )

        assert result.exit_code == 1
        assert "❌ Build failed" in result.stdout
        assert "Model not found" in result.stdout

    def test_build_command_no_quantization(self, runner, tmp_path):
        """Test build command with quantization disabled."""
        with patch.multiple(
            "src.cli.main",
            HuggingFaceModelLoader=MagicMock(),
            OnnxExporter=MagicMock(),
            create_bundle_from_pipeline=MagicMock(),
        ):
            runner.invoke(
                app, ["build", "gpt2", "--output", str(tmp_path), "--no-quantize"]
            )

            # Should not fail due to quantization parameter
            # Note: This test mainly checks CLI argument parsing

    @patch("src.cli.main.load_bundle")
    def test_info_command_success(self, mock_load_bundle, runner, tmp_path):
        """Test successful info command."""
        # Mock bundle with all required fields as strings (not MagicMocks)
        mock_bundle = MagicMock()
        mock_bundle.metadata.model_id = "gpt2"
        mock_bundle.metadata.architecture = "GPT2LMHeadModel"
        mock_bundle.metadata.target = "cpu"
        mock_bundle.metadata.preset = "balanced"
        mock_bundle.metadata.quantization_method = None  # No quantization for this test
        mock_bundle.metadata.quantization_alpha = None
        mock_bundle.metadata.original_size_mb = None
        mock_bundle.metadata.quantized_size_mb = None
        mock_bundle.metadata.compression_ratio = None
        mock_bundle.metadata.created_at.strftime.return_value = "2024-01-15 12:00:00"
        mock_bundle.metadata.alki_version = "0.1.0"
        mock_bundle.runtime_config.provider = "CPUExecutionProvider"
        mock_bundle.bundle_path = tmp_path
        mock_bundle.validate.return_value = []

        mock_load_bundle.return_value = mock_bundle

        bundle_path = tmp_path / "test_bundle"
        result = runner.invoke(app, ["info", str(bundle_path)])

        assert result.exit_code == 0
        assert "gpt2" in result.stdout
        assert "Bundle validation passed" in result.stdout
        mock_load_bundle.assert_called_once_with(bundle_path)

    @patch("src.cli.main.load_bundle")
    def test_info_command_failure(self, mock_load_bundle, runner):
        """Test info command with invalid bundle."""
        mock_load_bundle.side_effect = Exception("Invalid bundle")

        result = runner.invoke(app, ["info", "/nonexistent/bundle"])

        assert result.exit_code == 1
        assert "❌ Failed to load bundle" in result.stdout
        assert "Invalid bundle" in result.stdout

    @patch("src.cli.main.discover_bundles")
    def test_list_command_success(self, mock_discover_bundles, runner, tmp_path):
        """Test successful list command."""
        # Mock discovered bundles
        mock_bundle1 = MagicMock()
        mock_bundle1.metadata.model_id = "gpt2"
        mock_bundle1.metadata.target = "cpu"
        mock_bundle1.metadata.preset = "balanced"
        mock_bundle1.metadata.quantized_size_mb = 25
        mock_bundle1.metadata.compression_ratio = 0.25
        mock_bundle1.metadata.created_at.strftime.return_value = "2024-01-15 12:00"

        mock_bundle2 = MagicMock()
        mock_bundle2.metadata.model_id = "bert-base-uncased"
        mock_bundle2.metadata.target = "cpu"
        mock_bundle2.metadata.preset = "fast"
        mock_bundle2.metadata.original_size_mb = 200
        mock_bundle2.metadata.quantized_size_mb = None
        mock_bundle2.metadata.created_at.strftime.return_value = "2024-01-16 14:30"

        mock_discover_bundles.return_value = [mock_bundle1, mock_bundle2]

        result = runner.invoke(app, ["list", "--path", str(tmp_path)])

        assert result.exit_code == 0
        assert "gpt2" in result.stdout
        assert "bert-base-uncased" in result.stdout
        assert "Found 2 bundle(s)" in result.stdout
        mock_discover_bundles.assert_called_once_with(tmp_path, True)

    @patch("src.cli.main.discover_bundles")
    def test_list_command_no_bundles(self, mock_discover_bundles, runner, tmp_path):
        """Test list command with no bundles found."""
        mock_discover_bundles.return_value = []

        result = runner.invoke(app, ["list", "--path", str(tmp_path)])

        assert result.exit_code == 0
        assert "No bundles found" in result.stdout

    @patch("src.cli.main.discover_bundles")
    def test_list_command_verbose(self, mock_discover_bundles, runner, tmp_path):
        """Test list command with verbose output."""
        mock_bundle = MagicMock()
        mock_bundle.metadata.model_id = "gpt2"
        mock_bundle.metadata.target = "cpu"
        mock_bundle.metadata.preset = "balanced"
        mock_bundle.metadata.quantized_size_mb = 25
        mock_bundle.metadata.compression_ratio = 0.25
        mock_bundle.metadata.created_at.strftime.return_value = "2024-01-15 12:00"
        mock_bundle.metadata.quantization_method = "SmoothQuant W8A8"
        mock_bundle.bundle_path = tmp_path / "gpt2"

        mock_discover_bundles.return_value = [mock_bundle]

        result = runner.invoke(app, ["list", "--path", str(tmp_path), "--verbose"])

        assert result.exit_code == 0
        assert "SmoothQu" in result.stdout  # Quantization column (may be truncated)
        # Path is often truncated in Rich tables, just check for part of it
        assert "gpt2" in result.stdout  # Model name should be visible

    @pytest.mark.parametrize(
        "invalid_args,expected_error",
        [
            (["--alpha", "2.0"], "Invalid alpha"),  # Alpha should be 0.0-1.0
            # Target validation removed - only CPU supported, no validation needed
            (["--preset", "invalid_preset"], "Invalid preset"),
        ],
    )
    def test_build_command_parameter_validation(
        self, runner, tmp_path, invalid_args, expected_error
    ):
        """Test build command parameter validation."""
        result = runner.invoke(
            app,
            ["build", "gpt2"] + invalid_args + ["--output", str(tmp_path)],
        )

        assert result.exit_code != 0  # Should fail validation
