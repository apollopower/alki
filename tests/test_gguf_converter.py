"""
Unit tests for GGUFConverter.

Simple tests focused on core validation and mapping logic.
"""

import tempfile
from pathlib import Path

from src.converters.gguf.converter import GGUFConverter


class TestGGUFConverter:
    """Unit tests for GGUFConverter."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.converter = GGUFConverter()

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_converter_properties(self):
        """Test basic converter properties."""
        assert self.converter.name == "GGUF Converter (llama.cpp)"
        assert self.converter.target_format == "gguf"
        assert "Q4_K_M" in self.converter.supported_quantizations
        assert "LlamaForCausalLM" in self.converter.supported_architectures

    def test_quantization_mapping(self):
        """Test quantization type mapping."""
        assert self.converter._map_quantization_type("Q4_K_M") == "q4_k_m"
        assert self.converter._map_quantization_type("Q5_K_M") == "q5_k_m"
        assert self.converter._map_quantization_type("Q8_0") == "q8_0"

        # Test fallback for unknown types
        assert self.converter._map_quantization_type("UNKNOWN") == "unknown"

    def test_validate_output_valid_gguf(self):
        """Test GGUF validation with valid file."""
        # Create a file with GGUF magic bytes
        gguf_file = self.temp_dir / "valid.gguf"
        gguf_file.write_bytes(b"GGUF" + b"\x00" * 100)  # GGUF header + content

        assert self.converter.validate_output(gguf_file)

    def test_validate_output_invalid_gguf(self):
        """Test GGUF validation with invalid magic bytes."""
        # Create a file with wrong magic bytes
        invalid_file = self.temp_dir / "invalid.gguf"
        invalid_file.write_bytes(b"NOTGGUF\x00\x00\x01")

        assert not self.converter.validate_output(invalid_file)

    def test_validate_output_empty_file(self):
        """Test GGUF validation with empty file."""
        empty_file = self.temp_dir / "empty.gguf"
        empty_file.write_bytes(b"")

        assert not self.converter.validate_output(empty_file)

    def test_validate_output_nonexistent_file(self):
        """Test GGUF validation with nonexistent file."""
        nonexistent = self.temp_dir / "nonexistent.gguf"

        assert not self.converter.validate_output(nonexistent)

    def test_validate_output_wrong_extension(self):
        """Test GGUF validation with wrong file extension."""
        wrong_ext = self.temp_dir / "model.txt"
        wrong_ext.write_bytes(b"GGUF" + b"\x00" * 100)

        assert not self.converter.validate_output(wrong_ext)

    def test_get_unique_output_path(self):
        """Test unique output path generation."""
        # First call should return base name
        path1 = self.converter._get_unique_output_path(
            self.temp_dir, "test_model", "Q4_K_M"
        )
        expected1 = self.temp_dir / "test_model_q4_k_m.gguf"
        assert path1 == expected1

        # Create the file to simulate conflict
        path1.write_text("test")

        # Second call should return name with counter
        path2 = self.converter._get_unique_output_path(
            self.temp_dir, "test_model", "Q4_K_M"
        )
        expected2 = self.temp_dir / "test_model_q4_k_m_1.gguf"
        assert path2 == expected2
