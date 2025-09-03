"""Tests for the HuggingFace model loader."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.core.model_loader import HuggingFaceModelLoader


@pytest.fixture
def mock_hf_dependencies():
    """Fixture to mock all HuggingFace dependencies."""
    with (
        patch("src.core.model_loader.snapshot_download") as mock_download,
        patch("src.core.model_loader.AutoConfig.from_pretrained") as mock_config,
        patch("src.core.model_loader.AutoTokenizer.from_pretrained") as mock_tokenizer,
    ):

        # Setup default returns
        mock_download.return_value = "/mock/model/path"

        config_obj = MagicMock()
        config_obj.architectures = ["GPT2Model"]
        mock_config.return_value = config_obj

        mock_tokenizer.return_value = MagicMock()

        yield {
            "download": mock_download,
            "config": mock_config,
            "tokenizer": mock_tokenizer,
            "config_obj": config_obj,
        }


def test_prepare_downloads_and_returns_model_info(mock_hf_dependencies):
    """Test that prepare() correctly downloads model and returns expected structure."""
    loader = HuggingFaceModelLoader()
    result = loader.prepare("gpt2")

    # Verify structure
    assert result["model_id"] == "gpt2"
    assert result["local_path"] == Path("/mock/model/path")
    assert result["architecture"] == "GPT2Model"
    assert "config" in result
    assert "tokenizer" in result

    # Verify calls
    mock_hf_dependencies["download"].assert_called_once_with(repo_id="gpt2")
    mock_hf_dependencies["config"].assert_called_once_with("gpt2")
    mock_hf_dependencies["tokenizer"].assert_called_once_with("gpt2")


def test_prepare_handles_missing_architecture(mock_hf_dependencies):
    """Test graceful handling when model has no architecture field."""
    mock_hf_dependencies["config_obj"].architectures = None

    loader = HuggingFaceModelLoader()
    result = loader.prepare("test-model")

    assert result["architecture"] == "unknown"


def test_prepare_includes_size_estimation(mock_hf_dependencies):
    """Test that prepare() includes model size estimation."""
    # Setup config with typical GPT-2 parameters
    config = mock_hf_dependencies["config_obj"]
    config.architectures = ["GPT2LMHeadModel"]

    # Use patch to control getattr behavior for the config
    with patch("src.core.model_loader.getattr") as mock_getattr:

        def getattr_side_effect(obj, name, default=None):
            if name == "hidden_size":
                return 768
            elif name == "vocab_size":
                return 50257
            elif name == "num_hidden_layers":
                return 12
            elif name == "intermediate_size":
                return 3072
            elif name == "num_layers":
                return default  # Use default for num_layers fallback
            return default

        mock_getattr.side_effect = getattr_side_effect

        # Also ensure hasattr returns False for num_parameters
        with patch("src.core.model_loader.hasattr", return_value=False):
            loader = HuggingFaceModelLoader()
            result = loader.prepare("gpt2")

            # Should include size_mb estimation
            assert "size_mb" in result
            assert isinstance(result["size_mb"], float)
            assert result["size_mb"] > 0

            # For GPT-2 base, expect roughly 500-600MB
            assert 400 < result["size_mb"] < 800


def test_estimate_model_size_mb(mock_hf_dependencies):
    """Test model size estimation calculation logic."""
    loader = HuggingFaceModelLoader()

    # Test with GPT-2 config using patch to control getattr
    with (
        patch("src.core.model_loader.getattr") as mock_getattr,
        patch("src.core.model_loader.hasattr", return_value=False),
    ):

        def getattr_side_effect_gpt2(obj, name, default=None):
            values = {
                "hidden_size": 768,
                "vocab_size": 50257,
                "num_hidden_layers": 12,
                "intermediate_size": 3072,
            }
            return values.get(name, default)

        mock_getattr.side_effect = getattr_side_effect_gpt2
        config_gpt2 = MagicMock()

        size_mb = loader._estimate_model_size_mb(config_gpt2)
        assert isinstance(size_mb, float)
        assert size_mb > 0

    # Test with larger model (TinyLlama-like config)
    with (
        patch("src.core.model_loader.getattr") as mock_getattr_large,
        patch("src.core.model_loader.hasattr", return_value=False),
    ):

        def getattr_side_effect_large(obj, name, default=None):
            values = {
                "hidden_size": 2048,
                "vocab_size": 32000,
                "num_hidden_layers": 22,
                "intermediate_size": 5632,
            }
            return values.get(name, default)

        mock_getattr_large.side_effect = getattr_side_effect_large
        config_large = MagicMock()

        size_mb_large = loader._estimate_model_size_mb(config_large)
        assert size_mb_large > size_mb  # Larger model should have larger estimate
        assert size_mb_large > 2000  # TinyLlama should be > 2GB

    # Test fallback when estimation fails - create a config that raises an exception
    config_bad = MagicMock()
    with (
        patch("src.core.model_loader.getattr", side_effect=Exception("Config error")),
        patch("src.core.model_loader.hasattr", return_value=False),
    ):
        size_mb_fallback = loader._estimate_model_size_mb(config_bad)
        assert size_mb_fallback == 1000.0  # Default fallback
