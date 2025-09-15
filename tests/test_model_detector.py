"""
Unit tests for ModelDetector.

Simple tests focused on core detection logic without heavy mocking.
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

from src.core.model_detector import ModelDetector, ModelType


class TestModelDetector:
    """Unit tests for ModelDetector."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.detector = ModelDetector()

    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_detect_local_gguf_file(self):
        """Test detection of local GGUF files."""
        gguf_file = self.temp_dir / "model.gguf"
        gguf_file.write_text("test content")

        result = self.detector.detect_model_type(str(gguf_file))
        assert result == ModelType.LOCAL_GGUF

    def test_detect_local_non_gguf_file(self):
        """Test detection of local non-GGUF files."""
        txt_file = self.temp_dir / "model.txt"
        txt_file.write_text("test content")

        result = self.detector.detect_model_type(str(txt_file))
        assert result == ModelType.UNKNOWN

    def test_detect_nonexistent_local_file(self):
        """Test detection with nonexistent local file."""
        nonexistent = self.temp_dir / "nonexistent.gguf"

        # Should try to check as HuggingFace repo and fail
        with patch("src.core.model_detector.repo_exists", return_value=False):
            result = self.detector.detect_model_type(str(nonexistent))
            assert result == ModelType.UNKNOWN

    @patch("src.core.model_detector.repo_exists")
    @patch("src.core.model_detector.list_repo_files")
    def test_detect_hf_gguf_repo(self, mock_list_files, mock_repo_exists):
        """Test detection of HuggingFace GGUF repositories."""
        mock_repo_exists.return_value = True
        mock_list_files.return_value = ["README.md", "model.gguf", "config.json"]

        result = self.detector.detect_model_type("test/gguf-repo")
        assert result == ModelType.HF_GGUF

    @patch("src.core.model_detector.repo_exists")
    @patch("src.core.model_detector.list_repo_files")
    def test_detect_hf_pytorch_repo(self, mock_list_files, mock_repo_exists):
        """Test detection of HuggingFace PyTorch repositories."""
        mock_repo_exists.return_value = True
        mock_list_files.return_value = ["README.md", "config.json", "pytorch_model.bin"]

        result = self.detector.detect_model_type("test/pytorch-repo")
        assert result == ModelType.HF_PYTORCH

    @patch("src.core.model_detector.repo_exists")
    def test_detect_nonexistent_repo(self, mock_repo_exists):
        """Test detection with nonexistent repository."""
        mock_repo_exists.return_value = False

        result = self.detector.detect_model_type("test/nonexistent-repo")
        assert result == ModelType.UNKNOWN
