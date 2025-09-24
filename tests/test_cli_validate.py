"""
Tests for CLI Validate Command

Integration tests for the validate command functionality.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from typer.testing import CliRunner

from src.cli.main import app
from src.core.validator import ValidationResult


class TestCLIValidate:
    """Test CLI validate command integration"""

    def setup_method(self):
        """Set up test fixtures"""
        self.runner = CliRunner()

    def test_validate_local_file_without_benchmark(self):
        """Test validate command without benchmark flag"""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test.gguf"
            model_path.write_bytes(b"GGUF" + b"\x00" * 100)

            mock_result = ValidationResult(
                model_path=str(model_path),
                passed=True,
                load_time_ms=100.0,
                inference_time_ms=500.0,
                context_length=2048,
                vocab_size=32000,
                embedding_size=4096,
                inference_output="Test output",
            )

            with patch("src.cli.main.GGUFValidator") as mock_validator_class:
                mock_validator = MagicMock()
                mock_validator.validate_file.return_value = mock_result
                mock_validator_class.return_value = mock_validator

                result = self.runner.invoke(app, ["validate", str(model_path)])

                assert result.exit_code == 0
                assert "🎉 Validation completed successfully!" in result.stdout

                mock_validator.validate_file.assert_called_once()
                call_args = mock_validator.validate_file.call_args
                assert call_args[1]["benchmark"] is False

    def test_validate_local_file_with_benchmark(self):
        """Test validate command with --benchmark flag"""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test.gguf"
            model_path.write_bytes(b"GGUF" + b"\x00" * 100)

            mock_result = ValidationResult(
                model_path=str(model_path),
                passed=True,
                load_time_ms=100.0,
                inference_time_ms=1000.0,
                context_length=2048,
                vocab_size=32000,
                embedding_size=4096,
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                prompt_eval_rate=10.0,
                generation_rate=20.0,
                memory_usage_mb=600.0,
                inference_output="Test output",
            )

            with patch("src.cli.main.GGUFValidator") as mock_validator_class:
                mock_validator = MagicMock()
                mock_validator.validate_file.return_value = mock_result
                mock_validator_class.return_value = mock_validator

                result = self.runner.invoke(
                    app, ["validate", str(model_path), "--benchmark"]
                )

                assert result.exit_code == 0
                assert "🎉 Validation completed successfully!" in result.stdout

                mock_validator.validate_file.assert_called_once()
                call_args = mock_validator.validate_file.call_args
                assert call_args[1]["benchmark"] is True

    def test_validate_huggingface_with_benchmark(self):
        """Test validate HuggingFace repo with benchmark flag"""
        mock_result = ValidationResult(
            model_path="test/repo/*.gguf",
            passed=True,
            load_time_ms=100.0,
            inference_time_ms=1000.0,
            context_length=2048,
            vocab_size=32000,
            embedding_size=4096,
            prompt_tokens=15,
            completion_tokens=25,
            total_tokens=40,
            prompt_eval_rate=15.0,
            generation_rate=25.0,
            memory_usage_mb=700.0,
            inference_output="Test output",
        )

        with patch("src.cli.main.GGUFValidator") as mock_validator_class:
            mock_validator = MagicMock()
            mock_validator.validate_and_cleanup.return_value = mock_result
            mock_validator_class.return_value = mock_validator

            result = self.runner.invoke(
                app,
                [
                    "validate",
                    "test/repo",
                    "--filename",
                    "*.gguf",
                    "--benchmark",
                ],
            )

            assert result.exit_code == 0

            mock_validator.validate_and_cleanup.assert_called_once()
            call_args = mock_validator.validate_and_cleanup.call_args
            assert call_args[1]["benchmark"] is True

    def test_validate_benchmark_short_flag(self):
        """Test validate command with -b short flag"""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test.gguf"
            model_path.write_bytes(b"GGUF" + b"\x00" * 100)

            mock_result = ValidationResult(
                model_path=str(model_path),
                passed=True,
                load_time_ms=100.0,
                inference_time_ms=500.0,
                context_length=2048,
                vocab_size=32000,
                embedding_size=4096,
                generation_rate=20.0,
            )

            with patch("src.cli.main.GGUFValidator") as mock_validator_class:
                mock_validator = MagicMock()
                mock_validator.validate_file.return_value = mock_result
                mock_validator_class.return_value = mock_validator

                result = self.runner.invoke(app, ["validate", str(model_path), "-b"])

                assert result.exit_code == 0

                call_args = mock_validator.validate_file.call_args
                assert call_args[1]["benchmark"] is True

    def test_validate_output_includes_benchmark_section(self):
        """Test that CLI output includes benchmark section when present"""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test.gguf"
            model_path.write_bytes(b"GGUF" + b"\x00" * 100)

            mock_result = ValidationResult(
                model_path=str(model_path),
                passed=True,
                load_time_ms=100.0,
                inference_time_ms=1000.0,
                context_length=2048,
                vocab_size=32000,
                embedding_size=4096,
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                prompt_eval_rate=10.0,
                generation_rate=20.0,
                memory_usage_mb=600.0,
            )

            with patch("src.cli.main.GGUFValidator") as mock_validator_class:
                mock_validator = MagicMock()
                mock_validator.validate_file.return_value = mock_result
                mock_validator.print_result.side_effect = lambda r: print(
                    "WITH BENCHMARKS"
                )
                mock_validator_class.return_value = mock_validator

                result = self.runner.invoke(
                    app, ["validate", str(model_path), "--benchmark"]
                )

                assert "WITH BENCHMARKS" in result.stdout

    def test_validate_failed_validation(self):
        """Test validate command with failed validation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test.gguf"
            model_path.write_bytes(b"GGUF" + b"\x00" * 100)

            mock_result = ValidationResult(
                model_path=str(model_path),
                passed=False,
                load_time_ms=0,
                inference_time_ms=0,
                error="Model validation failed",
            )

            with patch("src.cli.main.GGUFValidator") as mock_validator_class:
                mock_validator = MagicMock()
                mock_validator.validate_file.return_value = mock_result
                mock_validator_class.return_value = mock_validator

                result = self.runner.invoke(app, ["validate", str(model_path)])

                assert result.exit_code == 1
                assert "❌ Validation failed!" in result.stdout

    def test_validate_nonexistent_file(self):
        """Test validate with nonexistent local file"""
        result = self.runner.invoke(app, ["validate", "/nonexistent/model.gguf"])

        assert result.exit_code == 1
        assert "For local files, ensure the file exists" in (
            result.stdout + result.stderr
        )

    def test_validate_custom_prompt_and_tokens(self):
        """Test validate with custom prompt and max tokens"""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test.gguf"
            model_path.write_bytes(b"GGUF" + b"\x00" * 100)

            mock_result = ValidationResult(
                model_path=str(model_path),
                passed=True,
                load_time_ms=100.0,
                inference_time_ms=500.0,
                context_length=2048,
                vocab_size=32000,
                embedding_size=4096,
            )

            with patch("src.cli.main.GGUFValidator") as mock_validator_class:
                mock_validator = MagicMock()
                mock_validator.validate_file.return_value = mock_result
                mock_validator_class.return_value = mock_validator

                result = self.runner.invoke(
                    app,
                    [
                        "validate",
                        str(model_path),
                        "--prompt",
                        "Custom test prompt",
                        "--max-tokens",
                        "100",
                        "--benchmark",
                    ],
                )

                assert result.exit_code == 0

                mock_validator_class.assert_called_once_with(
                    test_prompt="Custom test prompt"
                )

                call_args = mock_validator.validate_file.call_args
                assert call_args[1]["max_tokens"] == 100
                assert call_args[1]["benchmark"] is True
