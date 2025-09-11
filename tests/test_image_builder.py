"""
Tests for Image Builder

Focused tests covering essential functionality.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock
import pytest

from src.core.image_builder import ImageBuilder


class TestImageBuilder:
    """Test ImageBuilder essential functionality"""

    def create_test_bundle(self, bundle_path: Path) -> Path:
        """Create a minimal valid test bundle"""
        metadata_dir = bundle_path / "metadata"
        models_dir = bundle_path / "models"
        metadata_dir.mkdir(parents=True)
        models_dir.mkdir(parents=True)

        # Create model file
        model_file = models_dir / "test-model-q4_k_m.gguf"
        model_file.write_bytes(b"GGUF" + b"\x00" * 1000)

        # Create manifest
        manifest = {
            "artifacts": [
                {
                    "filename": "test-model-q4_k_m.gguf",
                    "quant": "Q4_K_M",
                    "uri": "./models/test-model-q4_k_m.gguf",
                    "sha256": "abc123",
                    "size": 1004,
                }
            ],
            "defaults": {"ctx": 2048},
        }

        manifest_path = metadata_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        return bundle_path

    def test_docker_availability_check(self):
        """Test Docker availability detection"""
        # Test when Docker is available
        with patch("src.core.image_builder.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            builder = ImageBuilder()
            assert builder.docker_client == "docker"

        # Test when Docker is not available
        with patch("src.core.image_builder.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("docker not found")
            with pytest.raises(RuntimeError, match="Docker not available"):
                ImageBuilder()

    def test_dockerfile_generation(self):
        """Test Dockerfile generation with real bundle"""
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "test-bundle"
            self.create_test_bundle(bundle_path)

            with patch("src.core.image_builder.subprocess.run") as mock_run:
                mock_run.return_value.returncode = 0
                builder = ImageBuilder()

                dockerfile = builder.generate_dockerfile(bundle_path)

                # Check essential Dockerfile elements
                assert "FROM ghcr.io/ggerganov/llama.cpp:server" in dockerfile
                assert '"-m", "/app/models/test-model-q4_k_m.gguf"' in dockerfile
                assert '"--ctx-size", "2048"' in dockerfile
                assert "EXPOSE 8080" in dockerfile
                assert "CMD" in dockerfile

    def test_dockerfile_custom_config(self):
        """Test Dockerfile with custom runtime configuration"""
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "test-bundle"
            self.create_test_bundle(bundle_path)

            with patch("src.core.image_builder.subprocess.run") as mock_run:
                mock_run.return_value.returncode = 0
                builder = ImageBuilder()

                runtime_config = {"ctx": 4096, "port": 9000}
                dockerfile = builder.generate_dockerfile(
                    bundle_path, base_image="alpine", runtime_config=runtime_config
                )

                assert "FROM ghcr.io/ggerganov/llama.cpp:server-alpine" in dockerfile
                assert '"--ctx-size", "4096"' in dockerfile
                assert '"--port", "9000"' in dockerfile
                assert "EXPOSE 9000" in dockerfile

    def test_error_handling(self):
        """Test common error scenarios"""
        with patch("src.core.image_builder.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            builder = ImageBuilder()

            with tempfile.TemporaryDirectory() as tmpdir:
                # Test missing bundle
                with pytest.raises(ValueError, match="Bundle manifest not found"):
                    builder.generate_dockerfile(Path(tmpdir) / "nonexistent")

                # Test bundle without artifacts
                bundle_path = Path(tmpdir) / "empty-bundle"
                bundle_path.mkdir()
                metadata_dir = bundle_path / "metadata"
                metadata_dir.mkdir()

                manifest_path = metadata_dir / "manifest.json"
                with open(manifest_path, "w") as f:
                    json.dump({"artifacts": []}, f)

                with pytest.raises(ValueError, match="Bundle has no model artifacts"):
                    builder.generate_dockerfile(bundle_path)

                # Test invalid base image
                valid_bundle = Path(tmpdir) / "valid-bundle"
                self.create_test_bundle(valid_bundle)

                with pytest.raises(ValueError, match="Unsupported base image"):
                    builder.generate_dockerfile(valid_bundle, base_image="invalid")

    def test_build_image_integration(self):
        """Test image building with realistic Docker interaction"""
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "test-bundle"
            self.create_test_bundle(bundle_path)

            # Mock Docker commands
            def mock_docker_cmd(cmd, **kwargs):
                result = Mock()
                result.returncode = 0
                result.stdout = ""
                result.stderr = ""

                if "version" in cmd:
                    result.stdout = "Docker version 20.10.0"
                elif "build" in cmd:
                    result.stdout = "Successfully built abc123"
                elif "images" in cmd and "--format" in cmd:
                    result.stdout = "500MB"

                return result

            with patch(
                "src.core.image_builder.subprocess.run", side_effect=mock_docker_cmd
            ):
                builder = ImageBuilder()
                result = builder.build_image(bundle_path, "test:latest")

                assert result.success
                assert result.image_tag == "test:latest"
                assert result.size_mb == 500.0
                assert result.build_time_seconds is not None
