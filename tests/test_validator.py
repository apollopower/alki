"""
Tests for GGUF Model Validator

Unit tests for the GGUFValidator class and benchmark functionality.
"""

import pytest
from unittest.mock import patch, MagicMock, Mock

from src.core.validator import GGUFValidator, ValidationResult


@pytest.fixture
def mock_llama():
    """Fixture to mock llama_cpp.Llama"""
    with patch("src.core.validator.Llama") as mock_llama_class:
        mock_model = MagicMock()

        mock_model.n_ctx.return_value = 2048
        mock_model.n_vocab.return_value = 32000
        mock_model.n_embd.return_value = 4096

        mock_response = {
            "choices": [{"text": "Test output from model"}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            },
        }
        mock_model.return_value = mock_response

        mock_llama_class.return_value = mock_model

        yield {
            "class": mock_llama_class,
            "instance": mock_model,
            "response": mock_response,
        }


@pytest.fixture
def mock_psutil():
    """Fixture to mock psutil for memory tracking"""
    with patch("src.core.validator.psutil.Process") as mock_process_class:
        mock_process = MagicMock()

        initial_memory = Mock()
        initial_memory.rss = 500 * 1024 * 1024

        post_load_memory = Mock()
        post_load_memory.rss = 1100 * 1024 * 1024

        mock_process.memory_info.side_effect = [initial_memory, post_load_memory]

        mock_process_class.return_value = mock_process

        yield {"class": mock_process_class, "instance": mock_process}


class TestValidationResult:
    """Test ValidationResult dataclass"""

    def test_validation_result_without_benchmark(self):
        """Test ValidationResult with no benchmark data"""
        result = ValidationResult(
            model_path="/test/model.gguf",
            passed=True,
            load_time_ms=100.0,
            inference_time_ms=500.0,
            context_length=2048,
            vocab_size=32000,
            embedding_size=4096,
        )

        assert result.prompt_tokens is None
        assert result.completion_tokens is None
        assert result.total_tokens is None
        assert result.prompt_eval_rate is None
        assert result.generation_rate is None
        assert result.memory_usage_mb is None

    def test_validation_result_with_benchmark(self):
        """Test ValidationResult with benchmark data"""
        result = ValidationResult(
            model_path="/test/model.gguf",
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

        assert result.prompt_tokens == 10
        assert result.completion_tokens == 20
        assert result.total_tokens == 30
        assert result.prompt_eval_rate == 10.0
        assert result.generation_rate == 20.0
        assert result.memory_usage_mb == 600.0

    def test_validation_result_to_dict(self):
        """Test ValidationResult serialization"""
        result = ValidationResult(
            model_path="/test/model.gguf",
            passed=True,
            load_time_ms=100.0,
            inference_time_ms=500.0,
            generation_rate=25.5,
        )

        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict["model_path"] == "/test/model.gguf"
        assert result_dict["generation_rate"] == 25.5


class TestGGUFValidator:
    """Test GGUFValidator class"""

    def test_validator_initialization(self):
        """Test validator initialization"""
        validator = GGUFValidator(test_prompt="Test prompt")
        assert validator.test_prompt == "Test prompt"
        assert validator.loader is not None

    @patch("src.core.validator.Path.exists")
    def test_validate_file_nonexistent(self, mock_exists):
        """Test validation fails for nonexistent file"""
        mock_exists.return_value = False

        validator = GGUFValidator()
        result = validator.validate_file("/nonexistent/model.gguf")

        assert result.passed is False
        assert "not found" in result.error

    @patch("src.core.validator.Path.exists")
    def test_validate_file_without_benchmark(self, mock_exists, mock_llama):
        """Test validate_file without benchmark flag"""
        mock_exists.return_value = True

        validator = GGUFValidator()
        result = validator.validate_file("/test/model.gguf", benchmark=False)

        assert result.passed is True
        assert result.load_time_ms > 0
        assert result.inference_time_ms > 0
        assert result.context_length == 2048
        assert result.vocab_size == 32000
        assert result.embedding_size == 4096
        assert result.inference_output == "Test output from model"

        assert result.prompt_tokens is None
        assert result.completion_tokens is None
        assert result.total_tokens is None
        assert result.prompt_eval_rate is None
        assert result.generation_rate is None
        assert result.memory_usage_mb is None

    @patch("src.core.validator.Path.exists")
    def test_validate_file_with_benchmark(self, mock_exists, mock_llama, mock_psutil):
        """Test validate_file with benchmark flag enabled"""
        mock_exists.return_value = True

        validator = GGUFValidator()
        result = validator.validate_file("/test/model.gguf", benchmark=True)

        assert result.passed is True
        assert result.load_time_ms > 0
        assert result.inference_time_ms > 0

        assert result.prompt_tokens == 10
        assert result.completion_tokens == 20
        assert result.total_tokens == 30

        assert result.generation_rate is not None
        assert result.generation_rate > 0
        assert result.prompt_eval_rate is not None
        assert result.prompt_eval_rate > 0

        assert result.memory_usage_mb is not None
        assert result.memory_usage_mb == 600.0

    @patch("src.core.validator.Path.exists")
    def test_benchmark_rate_calculations(self, mock_exists, mock_llama):
        """Test tokens/sec rate calculations are correct"""
        mock_exists.return_value = True

        mock_llama["instance"].return_value = {
            "choices": [{"text": "output"}],
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 100,
                "total_tokens": 150,
            },
        }

        validator = GGUFValidator()
        result = validator.validate_file("/test/model.gguf", benchmark=True)

        # Verify the calculations are correct given the actual inference time
        assert result.inference_time_ms > 0
        assert result.prompt_tokens == 50
        assert result.completion_tokens == 100

        # Verify rate calculations
        expected_generation_rate = (100 / result.inference_time_ms) * 1000
        expected_prompt_rate = (50 / result.inference_time_ms) * 1000

        assert result.generation_rate == pytest.approx(
            expected_generation_rate, rel=0.01
        )
        assert result.prompt_eval_rate == pytest.approx(expected_prompt_rate, rel=0.01)

    @patch("src.core.validator.Path.exists")
    def test_benchmark_handles_missing_usage(self, mock_exists, mock_llama):
        """Test benchmark gracefully handles missing usage data"""
        mock_exists.return_value = True

        mock_llama["instance"].return_value = {
            "choices": [{"text": "output"}],
        }

        validator = GGUFValidator()
        result = validator.validate_file("/test/model.gguf", benchmark=True)

        assert result.passed is True
        assert result.prompt_tokens == 0
        assert result.completion_tokens == 0
        assert result.total_tokens == 0

    def test_validate_huggingface_benchmark_passthrough(self, mock_llama, mock_psutil):
        """Test validate_huggingface passes benchmark parameter correctly"""
        validator = GGUFValidator()

        with patch.object(validator.loader, "prepareFromHuggingFace") as mock_prepare:
            mock_prepare.return_value = mock_llama["instance"]

            result = validator.validate_huggingface(
                "test/repo", "*.gguf", benchmark=True
            )

            assert result.passed is True
            assert result.prompt_tokens == 10
            assert result.completion_tokens == 20
            assert result.memory_usage_mb == 600.0

    def test_validate_and_cleanup_benchmark_passthrough(self, mock_llama):
        """Test validate_and_cleanup passes benchmark parameter through"""
        validator = GGUFValidator()

        with patch.object(validator, "validate_huggingface") as mock_validate:
            mock_validate.return_value = ValidationResult(
                model_path="test/repo/*.gguf",
                passed=True,
                load_time_ms=100.0,
                inference_time_ms=500.0,
                generation_rate=25.0,
            )

            with patch.object(validator, "cleanup_model_cache"):
                validator.validate_and_cleanup(
                    "test/repo", "*.gguf", benchmark=True, cleanup=False
                )

            mock_validate.assert_called_once()
            call_args, call_kwargs = mock_validate.call_args
            # benchmark is passed as 5th positional arg
            assert call_args[4] is True


class TestPrintResult:
    """Test print_result formatting"""

    def test_print_result_without_benchmark(self, capsys):
        """Test print output without benchmark data"""
        validator = GGUFValidator()

        result = ValidationResult(
            model_path="/test/model.gguf",
            passed=True,
            load_time_ms=100.5,
            inference_time_ms=500.3,
            context_length=2048,
            vocab_size=32000,
            embedding_size=4096,
            inference_output="Sample text",
        )

        validator.print_result(result)
        captured = capsys.readouterr()

        assert "GGUF MODEL VALIDATION RESULTS" in captured.out
        assert "WITH BENCHMARKS" not in captured.out
        assert "✅ PASSED" in captured.out
        assert "Load Time: 100.5ms" in captured.out
        assert "Inference Time: 500.3ms" in captured.out
        assert "Context Length: 2048" in captured.out
        assert "Performance Benchmarks:" not in captured.out

    def test_print_result_with_benchmark(self, capsys):
        """Test print output with benchmark data"""
        validator = GGUFValidator()

        result = ValidationResult(
            model_path="/test/model.gguf",
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
            memory_usage_mb=600.5,
            inference_output="Sample text",
        )

        validator.print_result(result)
        captured = capsys.readouterr()

        assert "WITH BENCHMARKS" in captured.out
        assert "Performance Benchmarks:" in captured.out
        assert "Memory Usage: 600.5 MB" in captured.out
        assert "Tokens: 10 prompt + 20 completion = 30 total" in captured.out
        assert "Generation Speed: 20.0 tokens/sec" in captured.out
        assert "Prompt Processing: 10.0 tokens/sec" in captured.out

    def test_print_result_failed_validation(self, capsys):
        """Test print output for failed validation"""
        validator = GGUFValidator()

        result = ValidationResult(
            model_path="/test/model.gguf",
            passed=False,
            load_time_ms=0,
            inference_time_ms=0,
            error="Model file corrupted",
        )

        validator.print_result(result)
        captured = capsys.readouterr()

        assert "❌ FAILED" in captured.out
        assert "Error: Model file corrupted" in captured.out
