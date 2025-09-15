"""
Tests for CLI Publish Command

Integration tests for the publish command functionality.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock
from typer.testing import CliRunner

from src.cli.main import app


class TestCLIPublish:
    """Test CLI publish command integration"""

    def create_test_bundle(self, bundle_path: Path) -> Path:
        """Create a minimal valid test bundle"""
        metadata_dir = bundle_path / "metadata"
        models_dir = bundle_path / "models"
        deploy_dir = bundle_path / "deploy"
        metadata_dir.mkdir(parents=True)
        models_dir.mkdir(parents=True)
        deploy_dir.mkdir(parents=True)

        model_file = models_dir / "test-model-q4_k_m.gguf"
        model_file.write_bytes(b"GGUF" + b"\x00" * 1000)

        manifest = {
            "name": "test-bundle",
            "version": "1.0.0",
            "created_at": "2024-01-01T12:00:00",
            "artifacts": [
                {
                    "filename": "test-model-q4_k_m.gguf",
                    "quant": "Q4_K_M",
                    "uri": "./models/test-bundle-test-model-q4_k_m.gguf",
                    "sha256": "abc123def456",
                    "size": 1004,
                }
            ],
            "defaults": {"ctx": 2048, "threads": "auto", "ngl": 0},
            "template": "chatml",
            "license": "apache-2.0",
        }

        manifest_path = metadata_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        return bundle_path

    def test_publish_command_validation(self):
        """Test publish command parameter validation"""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "test-bundle"
            self.create_test_bundle(bundle_path)

            # Test: No registry and no --local should fail
            result = runner.invoke(
                app, ["publish", str(bundle_path)], catch_exceptions=False
            )
            assert result.exit_code == 1
            # Error messages are sent to stderr in Typer, not stdout
            assert (
                "Either --registry must be specified or --local flag must be used"
                in (result.stdout + result.stderr)
            )

            # Test: Non-existent bundle should fail
            result = runner.invoke(
                app,
                ["publish", "/non/existent/path", "--local"],
                catch_exceptions=False,
            )
            assert result.exit_code == 1
            assert "Bundle path does not exist" in (result.stdout + result.stderr)

    def test_publish_missing_manifest(self):
        """Test publish with invalid bundle structure"""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create empty directory (no manifest)
            empty_bundle = Path(tmpdir) / "empty"
            empty_bundle.mkdir()

            result = runner.invoke(app, ["publish", str(empty_bundle), "--local"])
            assert result.exit_code == 1
            assert "Bundle manifest not found" in (result.stdout + result.stderr)
            assert "Run 'alki pack' first" in (result.stdout + result.stderr)

    @patch("src.core.registry_publisher.subprocess.run")
    @patch("src.core.bundle.Bundle.verify_bundle")
    def test_publish_local_only_success(self, mock_verify, mock_run):
        """Test successful local-only publish"""
        runner = CliRunner()

        # Mock bundle verification
        mock_verify.return_value = True

        # Mock Docker commands
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "Successfully built abc123"
        mock_run.return_value.stderr = ""

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "test-bundle"
            self.create_test_bundle(bundle_path)

            result = runner.invoke(
                app, ["publish", str(bundle_path), "--local", "--verbose"]
            )

            assert result.exit_code == 0
            assert "🏠 Local build - Processing bundle" in result.stdout
            assert "✅ Bundle built locally!" in result.stdout
            assert "📍 Local image: alki-local/test-bundle:latest" in result.stdout
            assert "🚀 To push to registry later:" in result.stdout

    @patch("src.core.registry_publisher.subprocess.run")
    @patch("src.core.bundle.Bundle.verify_bundle")
    def test_publish_with_registry_success(self, mock_verify, mock_run):
        """Test successful registry publish"""
        runner = CliRunner()

        # Mock bundle verification
        mock_verify.return_value = True

        # Mock successful Docker operations
        def mock_subprocess(*args, **kwargs):
            mock_result = Mock()
            mock_result.returncode = 0
            if "build" in str(args):
                mock_result.stdout = "Successfully built abc123"
            elif "push" in str(args):
                mock_result.stdout = "latest: digest: sha256:def456 size: 12345"
            else:
                mock_result.stdout = "Docker operation success"
            mock_result.stderr = ""
            return mock_result

        mock_run.side_effect = mock_subprocess

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "test-bundle"
            self.create_test_bundle(bundle_path)

            result = runner.invoke(
                app,
                [
                    "publish",
                    str(bundle_path),
                    "--registry",
                    "myregistry.com/bundles",
                    "--tag",
                    "v1.0.0",
                ],
            )

            assert result.exit_code == 0
            assert "🚀 Registry publish - Processing bundle" in result.stdout
            assert "✅ Bundle published successfully!" in result.stdout
            assert (
                "📍 Bundle URI: myregistry.com/bundles/test-bundle:v1.0.0"
                in result.stdout
            )
            assert "🚀 To deploy on Kubernetes:" in result.stdout
            assert "kubectl apply" in result.stdout

    @patch("src.core.registry_publisher.subprocess.run")
    @patch("src.core.bundle.Bundle.verify_bundle")
    def test_publish_with_output_manifest(self, mock_verify, mock_run):
        """Test publish with manifest output file"""
        runner = CliRunner()

        # Mock bundle verification and Docker
        mock_verify.return_value = True
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "Successfully built"
        mock_run.return_value.stderr = ""

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "test-bundle"
            self.create_test_bundle(bundle_path)

            manifest_file = Path(tmpdir) / "deployment.yaml"

            result = runner.invoke(
                app,
                [
                    "publish",
                    str(bundle_path),
                    "--local",
                    "--output-manifest",
                    str(manifest_file),
                ],
            )

            assert result.exit_code == 0
            assert f"📄 Deployment manifest saved: {manifest_file}" in result.stdout

            # Check manifest file was created
            assert manifest_file.exists()
            manifest_content = manifest_file.read_text()
            assert "apiVersion: v1" in manifest_content
            assert "kind: ConfigMap" in manifest_content
            assert "kind: Deployment" in manifest_content

    def test_publish_no_validate_flag(self):
        """Test --no-validate flag skips validation"""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "test-bundle"
            self.create_test_bundle(bundle_path)

            with patch("src.core.registry_publisher.subprocess.run") as mock_run:
                mock_run.return_value.returncode = 0
                mock_run.return_value.stdout = "Successfully built"

                result = runner.invoke(
                    app, ["publish", str(bundle_path), "--local", "--no-validate"]
                )

                # Should not show validation step
                assert "🔍 Validating bundle structure" not in result.stdout
                # But should still succeed
                assert result.exit_code == 0

    def test_publish_password_security_warning(self):
        """Test security warning for CLI password usage"""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "test-bundle"
            self.create_test_bundle(bundle_path)

            with patch("src.core.registry_publisher.subprocess.run") as mock_run:
                mock_run.return_value.returncode = 0
                mock_run.return_value.stdout = "Successfully built"

                result = runner.invoke(
                    app,
                    [
                        "publish",
                        str(bundle_path),
                        "--registry",
                        "test.com",
                        "--username",
                        "user",
                        "--password",
                        "secret123",
                    ],
                )

                assert (
                    "⚠️  Warning: Providing passwords via command line is insecure"
                    in (result.stdout + result.stderr)
                )
                assert "REGISTRY_USERNAME and REGISTRY_PASSWORD" in (
                    result.stdout + result.stderr
                )

    @patch("src.core.registry_publisher.subprocess.run")
    @patch("src.core.bundle.Bundle.verify_bundle")
    def test_publish_build_failure(self, mock_verify, mock_run):
        """Test handling of Docker build failures"""
        runner = CliRunner()

        # Mock bundle verification success but Docker build failure
        mock_verify.return_value = True
        mock_run.return_value.returncode = 1
        mock_run.return_value.stderr = "Docker build failed: syntax error"

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "test-bundle"
            self.create_test_bundle(bundle_path)

            result = runner.invoke(app, ["publish", str(bundle_path), "--local"])

            assert result.exit_code == 1
            assert "❌ Build error:" in (result.stdout + result.stderr)
            assert "Docker build failed" in (result.stdout + result.stderr)

    @patch("src.core.bundle.Bundle.verify_bundle")
    def test_publish_validation_failure(self, mock_verify):
        """Test handling of bundle validation failures"""
        runner = CliRunner()

        # Mock bundle verification failure
        mock_verify.return_value = False

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "test-bundle"
            self.create_test_bundle(bundle_path)

            result = runner.invoke(app, ["publish", str(bundle_path), "--local"])

            assert result.exit_code == 1
            assert "❌ Bundle validation failed!" in (result.stdout + result.stderr)

    def test_publish_custom_name_and_namespace(self):
        """Test custom bundle name and namespace parameters"""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "test-bundle"
            self.create_test_bundle(bundle_path)

            with patch("src.core.registry_publisher.subprocess.run") as mock_run:
                mock_run.return_value.returncode = 0
                mock_run.return_value.stdout = "Successfully built"

                with patch("src.core.bundle.Bundle.verify_bundle") as mock_verify:
                    mock_verify.return_value = True

                    result = runner.invoke(
                        app,
                        [
                            "publish",
                            str(bundle_path),
                            "--local",
                            "--name",
                            "custom-model",
                            "--namespace",
                            "production",
                        ],
                    )

                    assert result.exit_code == 0
                    assert "alki-local/custom-model:latest" in result.stdout

    def test_publish_verbose_logging(self):
        """Test verbose logging output"""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "test-bundle"
            self.create_test_bundle(bundle_path)

            with patch("src.core.registry_publisher.subprocess.run") as mock_run:
                mock_run.return_value.returncode = 0
                mock_run.return_value.stdout = "Successfully built"

                with patch("src.core.bundle.Bundle.verify_bundle") as mock_verify:
                    mock_verify.return_value = True

                    # Test with verbose flag
                    result = runner.invoke(
                        app, ["publish", str(bundle_path), "--local", "--verbose"]
                    )

                    assert result.exit_code == 0
                    # Verbose mode should show more detailed output
                    # (Actual detailed logging would be captured in logs, not stdout)

    def test_publish_help_command(self):
        """Test publish command help output"""
        runner = CliRunner()

        result = runner.invoke(app, ["publish", "--help"])

        assert result.exit_code == 0

        # Strip ANSI escape codes for consistent testing across environments
        import re

        clean_output = re.sub(r"\x1b\[[0-9;]*m", "", result.stdout)

        assert (
            "Publish bundle to container registry for fleet deployment" in clean_output
        )
        assert "--registry" in clean_output
        assert "--local" in clean_output
        assert "--username" in clean_output
        assert "--output-manifest" in clean_output
        assert "Examples:" in clean_output
        assert "alki publish" in clean_output
