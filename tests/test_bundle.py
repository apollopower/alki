"""
Tests for Bundle Management

Tests the bundle creation, manifest generation, and verification functionality.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from src.core.bundle import Bundle, BundleArtifact
from src.core.manifest import ManifestGenerator, ModelInfo


class TestBundle:
    """Test Bundle class functionality"""

    def test_bundle_initialization(self):
        """Test bundle initialization"""
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle = Bundle(Path(tmpdir), "test-model")

            assert bundle.name == "test-model"
            assert bundle.version is not None
            assert bundle.bundle_dir == Path(tmpdir) / "test-model"

    def test_bundle_name_sanitization(self):
        """Test bundle name sanitization"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test various name formats that should be sanitized
            test_cases = [
                ("Test_Model.gguf", "test-model-gguf"),
                ("QWEN_3.0.6B", "qwen-3-0-6b"),
                ("my_model.Q4_K_M", "my-model-q4-k-m"),
                ("Model.Name.With.Dots", "model-name-with-dots"),
                ("simple", "simple"),
            ]

            for input_name, expected_name in test_cases:
                bundle = Bundle(Path(tmpdir), input_name)
                assert bundle.name == expected_name
                assert bundle.bundle_dir == Path(tmpdir) / expected_name

    def test_bundle_structure_creation(self):
        """Test creation of bundle directory structure"""
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle = Bundle(Path(tmpdir), "test-model")
            bundle.create_structure()

            # Check all directories exist
            assert bundle.bundle_dir.exists()
            assert bundle.models_dir.exists()
            assert bundle.metadata_dir.exists()
            assert bundle.deploy_dir.exists()
            assert (bundle.deploy_dir / "systemd").exists()
            assert (bundle.deploy_dir / "docker").exists()

    def test_add_model(self):
        """Test adding a model to the bundle"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a fake model file
            model_path = Path(tmpdir) / "test.gguf"
            model_path.write_bytes(b"fake model content")

            bundle = Bundle(Path(tmpdir) / "output", "test-model")
            bundle.create_structure()

            artifact = bundle.add_model(model_path, "Q4_K_M")

            assert artifact.quant == "Q4_K_M"
            assert artifact.size == 18
            assert artifact.sha256 is not None
            # New behavior: filename preserves original name
            assert (bundle.models_dir / "test-model-test.gguf").exists()

    def test_create_manifest(self):
        """Test manifest creation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle = Bundle(Path(tmpdir), "test-model", version="1.0.0")
            bundle.create_structure()

            artifacts = [
                BundleArtifact(
                    quant="Q4_K_M",
                    uri="./models/test-q4_k_m.gguf",
                    sha256="abc123",
                    size=1000,
                    filename="test.gguf",
                )
            ]

            manifest = bundle.create_manifest(
                artifacts=artifacts,
                template="llama3",
                license="MIT",
                source_model="test/model",
                context_size=2048,
            )

            assert manifest.name == "test-model"
            assert manifest.version == "1.0.0"
            assert len(manifest.artifacts) == 1
            assert manifest.template == "llama3"
            assert manifest.defaults["ctx"] == 2048

            # Check manifest file was created
            manifest_path = bundle.metadata_dir / "manifest.json"
            assert manifest_path.exists()

            with open(manifest_path) as f:
                manifest_data = json.load(f)
                assert manifest_data["name"] == "test-model"

    def test_bundle_verification(self):
        """Test bundle verification"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a complete bundle
            model_path = Path(tmpdir) / "test.gguf"
            model_path.write_bytes(b"fake model content")

            bundle = Bundle(Path(tmpdir) / "output", "test-model")
            bundle.create_structure()

            artifact = bundle.add_model(model_path, "Q4_K_M")
            bundle.create_manifest([artifact])

            # Verification should pass
            assert bundle.verify_bundle() is True

    def test_bundle_verification_missing_file(self):
        """Test bundle verification with missing file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle = Bundle(Path(tmpdir), "test-model")
            bundle.create_structure()

            # Create manifest pointing to non-existent file
            artifacts = [
                BundleArtifact(
                    quant="Q4_K_M",
                    uri="./models/missing.gguf",
                    sha256="abc123",
                    size=1000,
                    filename="missing.gguf",
                )
            ]
            bundle.create_manifest(artifacts)

            # Verification should fail
            assert bundle.verify_bundle() is False


class TestManifestGenerator:
    """Test ManifestGenerator functionality"""

    def test_template_detection(self):
        """Test chat template detection"""
        generator = ManifestGenerator()

        assert generator.detect_chat_template("qwen3-0.6b") == "chatml"
        assert generator.detect_chat_template("llama-3.1-8b") == "llama3"
        assert generator.detect_chat_template("mistral-7b") == "mistral"
        assert generator.detect_chat_template("phi-3-mini") == "phi3"
        assert generator.detect_chat_template("unknown-model") == "chatml"  # default

    def test_generate_model_manifest(self):
        """Test model manifest generation"""
        generator = ManifestGenerator()

        artifacts = [
            {
                "quant": "Q4_K_M",
                "uri": "./models/test.gguf",
                "sha256": "abc123",
                "size": 1000,
            }
        ]

        model_info = ModelInfo(
            architecture="LlamaForCausalLM",
            context_length=4096,
            vocab_size=32000,
            embedding_size=4096,
            license="Apache-2.0",
        )

        manifest = generator.generate_model_manifest(
            name="test-model",
            version="1.0.0",
            artifacts=artifacts,
            model_info=model_info,
        )

        assert manifest["name"] == "test-model"
        assert manifest["version"] == "1.0.0"
        assert manifest["chat_template"] == "chatml"  # default
        assert manifest["defaults"]["ctx"] == 4096
        assert manifest["model_info"]["context_length"] == 4096
        assert manifest["model_info"]["license"] == "Apache-2.0"

    def test_generate_runtime_manifest(self):
        """Test runtime manifest generation"""
        generator = ManifestGenerator()

        model_args = {"ctx": 2048, "threads": 4, "ngl": 0}

        manifest = generator.generate_runtime_manifest(
            runtime="llama.cpp", model_args=model_args
        )

        assert manifest["runtime"] == "llama.cpp"
        assert manifest["server"]["port"] == 8080
        assert manifest["args"]["ctx"] == 2048
        assert "warnings" not in manifest

    def test_generate_runtime_manifest_no_context(self):
        """Test runtime manifest generation without context size"""
        generator = ManifestGenerator()

        manifest = generator.generate_runtime_manifest()

        assert "warnings" in manifest
        assert "Context size" in manifest["warnings"][0]

    def test_deployment_placeholder_systemd(self):
        """Test systemd deployment placeholder generation"""
        generator = ManifestGenerator()

        content = generator.create_deployment_placeholder(
            target="systemd",
            bundle_name="test-model",
            model_filename="test.gguf",
            context_size=2048,
            chat_template="llama3",
        )

        assert "[Service]" in content
        assert "test.gguf" in content
        assert "--ctx-size 2048" in content
        assert "--chat-format llama3" in content

    def test_deployment_placeholder_docker(self):
        """Test Docker deployment placeholder generation"""
        generator = ManifestGenerator()

        content = generator.create_deployment_placeholder(
            target="docker",
            bundle_name="test-model",
            model_filename="test.gguf",
            context_size=2048,
        )

        assert "FROM ghcr.io/ggml-org/llama.cpp:server" in content
        assert "LLAMA_ARG_CTX_SIZE=2048" in content
        assert "test.gguf" in content

    def test_deployment_placeholder_k8s(self):
        """Test Kubernetes deployment placeholder generation"""
        generator = ManifestGenerator()

        content = generator.create_deployment_placeholder(
            target="k3s",
            bundle_name="test-model",
            model_filename="test.gguf",
            context_size=2048,
            chat_template="llama3",
        )

        # Check for all K8s resources
        assert "kind: ConfigMap" in content
        assert "kind: PersistentVolumeClaim" in content
        assert "kind: Deployment" in content
        assert "kind: Service" in content

        # Check configuration values
        assert "test.gguf" in content
        assert "2048" in content
        assert "llama3" in content

        # Check proper K8s structure
        assert "apiVersion: v1" in content
        assert "apiVersion: apps/v1" in content
        assert "name: test-model" in content
        assert "ghcr.io/ggml-org/llama.cpp:server" in content

        # Check resource limits and health checks
        assert "resources:" in content
        assert "livenessProbe:" in content
        assert "readinessProbe:" in content

    def test_generate_sbom(self):
        """Test SBOM generation"""
        generator = ManifestGenerator()

        sbom = generator.generate_sbom(bundle_name="test-model", bundle_version="1.0.0")

        assert sbom["spdxVersion"] == "SPDX-2.3"
        assert sbom["name"] == "test-model-sbom"
        assert len(sbom["packages"]) == 2  # bundle + llama.cpp
        assert len(sbom["relationships"]) == 2

    @patch("src.core.manifest.Llama")
    def test_extract_model_capabilities(self, mock_llama):
        """Test model capability extraction"""
        # Mock the Llama model
        mock_model = Mock()
        mock_model.n_ctx_train.return_value = 4096
        mock_model.n_vocab.return_value = 32000
        mock_model.n_embd.return_value = 4096
        mock_llama.return_value = mock_model

        generator = ManifestGenerator()

        with tempfile.NamedTemporaryFile(suffix=".gguf") as tmpfile:
            capabilities = generator.extract_model_capabilities(Path(tmpfile.name))

            assert capabilities is not None
            assert capabilities["context_length"] == 4096
            assert capabilities["vocab_size"] == 32000
            assert capabilities["embedding_size"] == 4096


class TestEndToEnd:
    """End-to-end bundle creation tests"""

    def test_complete_bundle_creation(self):
        """Test complete bundle creation workflow"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a fake model
            model_path = Path(tmpdir) / "model.gguf"
            model_path.write_bytes(b"fake model content for testing")

            # Create bundle
            bundle = Bundle(Path(tmpdir) / "output", "test-model", "1.0.0")
            bundle.create_structure()

            # Add model
            artifact = bundle.add_model(model_path, "Q4_K_M")

            # Create manifests
            bundle.create_manifest(
                artifacts=[artifact],
                template="llama3",
                license="MIT",
                context_size=2048,
            )

            bundle.create_runtime_manifest()
            bundle.create_sbom()
            bundle.add_readme("test-model", ["Q4_K_M"])
            bundle.add_license("MIT License\n\nTest license content")

            # Add deployment placeholders
            generator = ManifestGenerator()

            systemd_config = generator.create_deployment_placeholder(
                "systemd", "test-model", "test-model-q4_k_m.gguf", 2048
            )
            (bundle.deploy_dir / "systemd" / "alki-test-model.service").write_text(
                systemd_config
            )

            docker_config = generator.create_deployment_placeholder(
                "docker", "test-model", "test-model-q4_k_m.gguf", 2048
            )
            (bundle.deploy_dir / "docker" / "Dockerfile").write_text(docker_config)

            # Verify bundle
            assert bundle.verify_bundle() is True

            # Check all expected files exist
            assert (bundle.metadata_dir / "manifest.json").exists()
            assert (bundle.metadata_dir / "runtime.json").exists()
            assert (bundle.metadata_dir / "sbom.spdx.json").exists()
            assert (bundle.metadata_dir / "README.md").exists()
            assert (bundle.metadata_dir / "LICENSE.txt").exists()
            assert (bundle.deploy_dir / "systemd" / "alki-test-model.service").exists()
            assert (bundle.deploy_dir / "docker" / "Dockerfile").exists()

            # Get bundle info
            info = bundle.get_info()
            assert info["name"] == "test-model"
            assert info["version"] == "1.0.0"
            assert info["artifacts"] == 1

    def test_filename_preservation_prevents_misleading_names(self):
        """Test that filenames preserve actual quantization info"""
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle = Bundle(Path(tmpdir), "mymodel")
            bundle.create_structure()

            # Test different scenarios
            test_cases = [
                ("model-q8_0.gguf", "Q4_K_M"),  # Q8_0 model, requesting Q4_K_M
                ("model-q5_k_m.gguf", "Q8_0"),  # Q5_K_M model, requesting Q8_0
                ("model-f16.gguf", "Q4_0"),  # F16 model, requesting Q4_0
            ]

            for original_filename, requested_quant in test_cases:
                model_file = Path(tmpdir) / original_filename
                model_file.write_bytes(b"GGUF fake model data")

                artifact = bundle.add_model(model_file, quantization=requested_quant)

                # Original filename should be preserved in artifact.filename
                assert artifact.filename == original_filename

                # Bundle filename should include original name
                expected_bundle_name = f"mymodel-{original_filename}"
                assert expected_bundle_name in artifact.uri

                # Quantization mismatch should be clear in metadata
                assert artifact.quant == requested_quant

                # Actual file should exist with preserved name
                bundle_file_path = bundle.models_dir / expected_bundle_name
                assert bundle_file_path.exists()
