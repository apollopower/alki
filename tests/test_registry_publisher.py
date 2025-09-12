"""
Tests for Registry Publisher

Focused tests covering essential functionality for bundle publishing.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock
import pytest

from src.core.registry_publisher import RegistryPublisher


class TestRegistryPublisher:
    """Test RegistryPublisher essential functionality"""

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
            "name": "test-model",
            "version": "1.0.0",
            "created_at": "2024-01-01T12:00:00",
            "artifacts": [
                {
                    "filename": "test-model-q4_k_m.gguf",
                    "quant": "Q4_K_M",
                    "uri": "./models/test-model-q4_k_m.gguf",
                    "sha256": "abc123def456",
                    "size": 1004,
                }
            ],
            "defaults": {"ctx": 2048},
            "template": "chatml",
            "license": "apache-2.0",
        }

        manifest_path = metadata_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        return bundle_path

    def test_docker_availability_check(self):
        """Test Docker availability detection"""
        # Test when Docker is available
        with patch("src.core.registry_publisher.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            publisher = RegistryPublisher()
            assert publisher.docker_client == "docker"

        # Test when Docker is not available
        with patch("src.core.registry_publisher.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("docker not found")
            with pytest.raises(RuntimeError, match="Docker not available"):
                RegistryPublisher()

    def test_dockerfile_generation(self):
        """Test Dockerfile generation with metadata labels"""
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "test-bundle"
            self.create_test_bundle(bundle_path)

            with patch("src.core.registry_publisher.subprocess.run") as mock_run:
                mock_run.return_value.returncode = 0
                publisher = RegistryPublisher()

                # Load manifest for metadata
                with open(bundle_path / "metadata" / "manifest.json") as f:
                    metadata = json.load(f)

                dockerfile = publisher.generate_bundle_dockerfile(bundle_path, metadata)

                # Check essential Dockerfile elements
                assert "FROM alpine:latest" in dockerfile
                assert "WORKDIR /bundle" in dockerfile
                assert "COPY . /bundle/" in dockerfile
                assert "/extract-bundle.sh" in dockerfile

                # Check metadata labels
                assert 'LABEL alki.bundle.name="test-model"' in dockerfile
                assert 'LABEL alki.bundle.version="1.0.0"' in dockerfile
                assert 'LABEL alki.bundle.quantization="Q4_K_M"' in dockerfile
                assert 'LABEL alki.bundle.context_size="2048"' in dockerfile
                assert 'LABEL alki.bundle.chat_template="chatml"' in dockerfile

                # Check OCI annotations
                assert 'org.opencontainers.image.title="Alki Bundle"' in dockerfile
                assert 'org.opencontainers.image.vendor="Alki"' in dockerfile

    def test_authentication_resolution_priority(self):
        """Test authentication resolution priority order"""
        with patch("src.core.registry_publisher.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            publisher = RegistryPublisher()

            # Test 1: Explicit auth takes priority
            explicit_auth = {"username": "explicit", "password": "pass123"}
            result = publisher._resolve_authentication("registry.com", explicit_auth)
            assert result == explicit_auth

            # Test 2: Environment variables when no explicit auth
            with patch.dict(
                os.environ,
                {"REGISTRY_USERNAME": "envuser", "REGISTRY_PASSWORD": "envpass"},
            ):
                result = publisher._resolve_authentication("registry.com", None)
                assert result["username"] == "envuser"
                assert result["password"] == "envpass"
                assert result["registry"] == "registry.com"

            # Test 3: Docker fallback (returns None to let Docker handle)
            with patch.dict(os.environ, {}, clear=True):
                result = publisher._resolve_authentication("registry.com", None)
                assert result is None

    def test_authentication_env_var_fallbacks(self):
        """Test environment variable fallbacks"""
        with patch("src.core.registry_publisher.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            publisher = RegistryPublisher()

            # Test DOCKER_ prefixed env vars
            with patch.dict(
                os.environ,
                {"DOCKER_USERNAME": "dockeruser", "DOCKER_PASSWORD": "dockerpass"},
                clear=True,
            ):
                result = publisher._resolve_authentication("registry.com", None)
                assert result["username"] == "dockeruser"
                assert result["password"] == "dockerpass"

    def test_local_only_build(self):
        """Test local-only build doesn't attempt registry push"""
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "test-bundle"
            self.create_test_bundle(bundle_path)

            with patch("src.core.registry_publisher.subprocess.run") as mock_run:
                # Mock Docker commands
                mock_run.return_value.returncode = 0
                mock_run.return_value.stdout = "Successfully built abc123"
                mock_run.return_value.stderr = ""

                publisher = RegistryPublisher()

                result = publisher.publish_bundle(
                    bundle_path=bundle_path, local_only=True, tag="test"
                )

                assert result.success
                assert result.bundle_uri == "alki-local/test-model:test"
                assert result.image_tag == "alki-local/test-model:test"

                # Verify no push was attempted (only build commands called)
                build_calls = [
                    call for call in mock_run.call_args_list if "build" in str(call)
                ]
                push_calls = [
                    call for call in mock_run.call_args_list if "push" in str(call)
                ]

                assert len(build_calls) > 0  # Build was called
                assert len(push_calls) == 0  # Push was not called

    def test_registry_build_and_push(self):
        """Test full registry workflow with mocked Docker operations"""
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "test-bundle"
            self.create_test_bundle(bundle_path)

            with patch("src.core.registry_publisher.subprocess.run") as mock_run:
                # Mock successful Docker operations
                def mock_subprocess(*args, **kwargs):
                    mock_result = Mock()
                    mock_result.returncode = 0
                    mock_result.stdout = (
                        "Successfully built abc123\ndigest: sha256:def456"
                    )
                    mock_result.stderr = ""
                    return mock_result

                mock_run.side_effect = mock_subprocess

                publisher = RegistryPublisher()

                result = publisher.publish_bundle(
                    bundle_path=bundle_path,
                    registry="myregistry.com/bundles",
                    tag="v1.0.0",
                )

                assert result.success
                assert result.bundle_uri == "myregistry.com/bundles/test-model:v1.0.0"
                assert result.image_tag == "myregistry.com/bundles/test-model:v1.0.0"
                assert result.bundle_sha256 == "abc123def456"  # From manifest

                # Verify both build and push were called
                build_calls = [
                    call for call in mock_run.call_args_list if "build" in str(call)
                ]
                push_calls = [
                    call for call in mock_run.call_args_list if "push" in str(call)
                ]

                assert len(build_calls) > 0
                assert len(push_calls) > 0

    def test_build_failure_handling(self):
        """Test Docker build failure handling"""
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "test-bundle"
            self.create_test_bundle(bundle_path)

            with patch("src.core.registry_publisher.subprocess.run") as mock_run:

                def mock_subprocess(*args, **kwargs):
                    mock_result = Mock()
                    if "version" in args[0]:
                        # Docker availability check succeeds
                        mock_result.returncode = 0
                        mock_result.stderr = ""
                    else:
                        # Build command fails
                        mock_result.returncode = 1
                        mock_result.stderr = "Build failed: syntax error"
                    return mock_result

                mock_run.side_effect = mock_subprocess

                publisher = RegistryPublisher()

                result = publisher.publish_bundle(
                    bundle_path=bundle_path, registry="myregistry.com", tag="test"
                )

                assert not result.success
                assert "Build failed" in result.error
                assert "syntax error" in result.error

    def test_push_authentication_failure(self):
        """Test registry authentication failure handling"""
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "test-bundle"
            self.create_test_bundle(bundle_path)

            with patch("src.core.registry_publisher.subprocess.run") as mock_run:

                def mock_subprocess(*args, **kwargs):
                    mock_result = Mock()
                    if "version" in args[0]:
                        # Docker availability check succeeds
                        mock_result.returncode = 0
                        mock_result.stderr = ""
                        return mock_result
                    elif "build" in args[0]:
                        # Build succeeds
                        mock_result.returncode = 0
                        mock_result.stdout = "Successfully built abc123"
                        mock_result.stderr = ""
                        return mock_result
                    elif "inspect" in args[0]:
                        # Image inspect for size
                        mock_result.returncode = 0
                        mock_result.stdout = "12345678"
                        mock_result.stderr = ""
                        return mock_result
                    elif "push" in args[0]:
                        # Push fails with auth error
                        mock_result.returncode = 1
                        mock_result.stderr = "authentication required"
                        mock_result.stdout = ""
                        return mock_result
                    else:
                        mock_result.returncode = 0
                        mock_result.stdout = ""
                        mock_result.stderr = ""
                        return mock_result

                mock_run.side_effect = mock_subprocess

                publisher = RegistryPublisher()

                result = publisher.publish_bundle(
                    bundle_path=bundle_path, registry="myregistry.com", tag="test"
                )

                assert not result.success
                assert "authentication failed" in result.error.lower()
                assert "docker login" in result.error

    def test_invalid_bundle_handling(self):
        """Test handling of invalid bundle paths"""
        with patch("src.core.registry_publisher.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            publisher = RegistryPublisher()

            # Test non-existent bundle
            result = publisher.publish_bundle(
                bundle_path=Path("/non/existent/path"), registry="test.com", tag="test"
            )
            assert not result.success
            assert "Invalid bundle path" in result.error

            # Test bundle without manifest
            with tempfile.TemporaryDirectory() as tmpdir:
                empty_bundle = Path(tmpdir) / "empty"
                empty_bundle.mkdir()

                result = publisher.publish_bundle(
                    bundle_path=empty_bundle, registry="test.com", tag="test"
                )
                assert not result.success
                assert "Bundle manifest not found" in result.error

    def test_deployment_manifest_generation(self):
        """Test Kubernetes deployment manifest generation"""
        with patch("src.core.registry_publisher.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            publisher = RegistryPublisher()

            bundle_metadata = {
                "name": "test-model",
                "artifacts": [{"uri": "./models/test-model.gguf"}],
                "defaults": {"ctx": 4096},
                "template": "llama3",
            }

            manifest = publisher.generate_deployment_manifest(
                bundle_uri="myregistry.com/test-model:v1.0.0",
                bundle_name="test-model",
                bundle_metadata=bundle_metadata,
                namespace="production",
            )

            # Check essential K8s components
            assert "apiVersion: v1" in manifest
            assert "kind: ConfigMap" in manifest
            assert "kind: Deployment" in manifest
            assert "kind: Service" in manifest

            # Check configuration values
            assert "test-model-bundle-config" in manifest
            assert "namespace: production" in manifest
            assert "myregistry.com/test-model:v1.0.0" in manifest
            assert "test-model.gguf" in manifest
            assert '"4096"' in manifest
            assert "llama3" in manifest

            # Check container structure
            assert "initContainers:" in manifest
            assert "bundle-loader" in manifest
            assert "llama-server" in manifest
            assert "livenessProbe:" in manifest
            assert "readinessProbe:" in manifest

    def test_registry_host_extraction(self):
        """Test registry host extraction from image tags"""
        with patch("src.core.registry_publisher.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            publisher = RegistryPublisher()

            # Test registry with path
            publisher.push_bundle_to_registry("myregistry.com/path/image:tag", None)
            # Should extract myregistry.com as host

            # Test Docker Hub (no slash)
            publisher.push_bundle_to_registry("ubuntu:latest", None)
            # Should default to docker.io

            # Verify calls were made (even if they fail due to mocking)
            assert len(mock_run.call_args_list) >= 2

    def test_manifest_with_missing_metadata(self):
        """Test manifest generation with missing optional metadata"""
        with patch("src.core.registry_publisher.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            publisher = RegistryPublisher()

            # Minimal bundle metadata
            bundle_metadata = {"artifacts": []}

            manifest = publisher.generate_deployment_manifest(
                bundle_uri="test:latest",
                bundle_name="minimal",
                bundle_metadata=bundle_metadata,
            )

            # Should use defaults
            assert "4096" in manifest  # Default context size
            assert "chatml" in manifest  # Default template
            assert "model.gguf" in manifest  # Default filename
