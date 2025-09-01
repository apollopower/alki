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
