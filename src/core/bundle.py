"""
Bundle Management for Alki

Handles creation and management of deployment bundles containing GGUF models,
manifests, and metadata for edge deployment.
"""

import hashlib
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

# Constants
HASH_CHUNK_SIZE = 262144  # 256KB chunks for efficient processing of large model files
DEFAULT_CONTEXT_SIZE = (
    4096  # Default context size for runtime manifest when not specified
)


@dataclass
class BundleArtifact:
    """Represents a single artifact in the bundle"""

    quant: str  # Quantization profile (Q4_K_M, Q5_K_M, Q8_0)
    uri: str  # Relative path within bundle
    sha256: str  # SHA256 hash of the file
    size: int  # Size in bytes
    filename: str  # Original filename


@dataclass
class BundleManifest:
    """Bundle metadata and manifest"""

    name: str
    version: str
    created_at: str
    artifacts: List[BundleArtifact]
    defaults: Dict[str, Any]
    template: Optional[str] = None
    license: Optional[str] = None
    source_model: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        # Convert artifact dataclasses to dicts
        data["artifacts"] = [asdict(a) for a in self.artifacts]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BundleManifest":
        """Create from dictionary"""
        artifacts = [BundleArtifact(**a) for a in data.get("artifacts", [])]
        data["artifacts"] = artifacts
        return cls(**data)


class Bundle:
    """Manages deployment bundle creation and structure"""

    def __init__(self, output_dir: Path, name: str, version: Optional[str] = None):
        """
        Initialize bundle

        Args:
            output_dir: Base output directory for bundle
            name: Bundle name (will be sanitized for consistency)
            version: Bundle version (auto-generated if not provided)
        """
        self.name = self._sanitize_name(name)
        self.version = version or self._generate_version()
        self.bundle_dir = output_dir / self.name
        self.models_dir = self.bundle_dir / "models"
        self.metadata_dir = self.bundle_dir / "metadata"
        self.deploy_dir = self.bundle_dir / "deploy"

        logger.info(f"Initializing bundle: {self.name} v{self.version}")

    def _sanitize_name(self, name: str) -> str:
        """
        Sanitize a name for use as a bundle name.

        Converts to lowercase and replaces underscores and dots with hyphens
        for consistent, URL-safe bundle names.

        Args:
            name: Raw name to sanitize

        Returns:
            Sanitized bundle name
        """
        return name.lower().replace("_", "-").replace(".", "-")

    def _generate_version(self) -> str:
        """Generate version string based on current date"""
        return datetime.now().strftime("%Y-%m-%d.1")

    def create_structure(self) -> None:
        """Create bundle directory structure"""
        logger.info(f"Creating bundle structure at {self.bundle_dir}")

        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.deploy_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for deployment configs
        (self.deploy_dir / "systemd").mkdir(exist_ok=True)
        (self.deploy_dir / "k3s").mkdir(exist_ok=True)
        (self.deploy_dir / "docker").mkdir(exist_ok=True)

        logger.debug(f"Created bundle directories: {list(self.bundle_dir.iterdir())}")

    def add_model(
        self, model_path: Path, quantization: str = "Q4_K_M"
    ) -> BundleArtifact:
        """
        Add a GGUF model to the bundle

        Args:
            model_path: Path to GGUF model file
            quantization: Quantization profile name

        Returns:
            BundleArtifact with model metadata
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Generate target filename
        target_name = f"{self.name}-{quantization.lower()}.gguf"
        target_path = self.models_dir / target_name

        logger.info(f"Adding model: {model_path.name} -> {target_name}")

        # Copy model file
        shutil.copy2(model_path, target_path)

        # Calculate SHA256 hash
        sha256_hash = self._calculate_sha256(target_path)

        # Get file size
        file_size = target_path.stat().st_size

        # Create artifact record
        artifact = BundleArtifact(
            quant=quantization,
            uri=f"./models/{target_name}",
            sha256=sha256_hash,
            size=file_size,
            filename=model_path.name,
        )

        logger.debug(f"Added model artifact: {artifact}")
        return artifact

    def _calculate_sha256(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file"""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            # Read in chunks to handle large files
            for chunk in iter(lambda: f.read(HASH_CHUNK_SIZE), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def create_manifest(
        self,
        artifacts: List[BundleArtifact],
        template: Optional[str] = None,
        license: Optional[str] = None,
        source_model: Optional[str] = None,
        context_size: int = 4096,
        threads: str = "auto",
    ) -> BundleManifest:
        """
        Create bundle manifest

        Args:
            artifacts: List of bundle artifacts
            template: Chat template name
            license: Model license
            source_model: Source model identifier
            context_size: Default context window size
            threads: Default thread configuration

        Returns:
            BundleManifest object
        """
        manifest = BundleManifest(
            name=self.name,
            version=self.version,
            created_at=datetime.now().isoformat(),
            artifacts=artifacts,
            defaults={
                "ctx": context_size,
                "threads": threads,
                "ngl": 0,  # Number of GPU layers (0 for CPU-only by default)
            },
            template=template,
            license=license,
            source_model=source_model,
        )

        # Write manifest to file
        manifest_path = self.metadata_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest.to_dict(), f, indent=2)

        logger.info(f"Created manifest: {manifest_path}")
        return manifest

    def create_runtime_manifest(
        self,
        runtime: str = "llama.cpp",
        host: str = "0.0.0.0",
        port: int = 8080,
        api: bool = True,
        context_size: int = DEFAULT_CONTEXT_SIZE,
    ) -> Dict[str, Any]:
        """
        Create runtime configuration manifest

        Args:
            runtime: Runtime engine name
            host: Server host
            port: Server port
            api: Enable API mode
            context_size: Context window size (defaults to DEFAULT_CONTEXT_SIZE)

        Returns:
            Runtime manifest dictionary
        """
        runtime_manifest = {
            "runtime": runtime,
            "server": {"host": host, "port": port, "api": api},
            "args": {"ctx": context_size, "threads": "auto", "ngl": 0},
            "health": {"path": "/v1/models", "timeout_s": 5},
        }

        # Write runtime manifest
        runtime_path = self.metadata_dir / "runtime.json"
        with open(runtime_path, "w") as f:
            json.dump(runtime_manifest, f, indent=2)

        logger.info(f"Created runtime manifest: {runtime_path}")
        return runtime_manifest

    def create_sbom(self, dependencies: Optional[List[Dict[str, str]]] = None) -> None:
        """
        Create Software Bill of Materials (SBOM) in SPDX format

        Args:
            dependencies: List of dependencies with name and version
        """
        sbom = {
            "spdxVersion": "SPDX-2.3",
            "dataLicense": "CC0-1.0",
            "SPDXID": "SPDXRef-DOCUMENT",
            "name": f"{self.name}-sbom",
            "documentNamespace": f"https://alki.ai/sbom/{self.name}/{self.version}",
            "creationInfo": {
                "created": datetime.now().isoformat(),
                "creators": ["Tool: alki"],
            },
            "packages": [
                {
                    "SPDXID": "SPDXRef-Package",
                    "name": self.name,
                    "downloadLocation": "NOASSERTION",
                    "filesAnalyzed": False,
                    "version": self.version,
                }
            ],
        }

        # Add dependencies if provided
        if dependencies:
            for i, dep in enumerate(dependencies):
                sbom["packages"].append(
                    {
                        "SPDXID": f"SPDXRef-Dependency-{i}",
                        "name": dep.get("name", "unknown"),
                        "version": dep.get("version", "unknown"),
                        "downloadLocation": "NOASSERTION",
                        "filesAnalyzed": False,
                    }
                )

        # Write SBOM
        sbom_path = self.metadata_dir / "sbom.spdx.json"
        with open(sbom_path, "w") as f:
            json.dump(sbom, f, indent=2)

        logger.info(f"Created SBOM: {sbom_path}")

    def add_readme(self, model_name: str, quantization_profiles: List[str]) -> None:
        """
        Create README.md for the bundle

        Args:
            model_name: Name of the model
            quantization_profiles: List of quantization profiles included
        """
        readme_content = f"""# {self.name} Bundle

## Quick Start

This bundle contains the {model_name} model optimized for edge deployment.

### Included Quantization Profiles
{chr(10).join(f"- {q}" for q in quantization_profiles)}

### Running with llama-server

```bash
llama-server \\
  -m ./models/{self.name}-q4_k_m.gguf \\
  --host 0.0.0.0 \\
  --port 8080 \\
  --ctx-size 4096
```

### API Usage

```bash
curl http://localhost:8080/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{{
    "model": "{self.name}",
    "messages": [{{"role": "user", "content": "Hello!"}}],
    "max_tokens": 100
  }}'
```

## Bundle Contents

- `models/` - GGUF model files
- `metadata/` - Manifests and metadata
- `deploy/` - Deployment configurations

## Version

Bundle Version: {self.version}
Created: {datetime.now().isoformat()}

---
Generated by Alki 🌊
"""

        readme_path = self.metadata_dir / "README.md"
        with open(readme_path, "w") as f:
            f.write(readme_content)

        logger.info(f"Created README: {readme_path}")

    def add_license(self, license_text: str) -> None:
        """
        Add license file to bundle

        Args:
            license_text: License text content
        """
        license_path = self.metadata_dir / "LICENSE.txt"
        with open(license_path, "w") as f:
            f.write(license_text)

        logger.info(f"Added license: {license_path}")

    def verify_bundle(self) -> bool:
        """
        Verify bundle integrity

        Returns:
            True if bundle is valid, False otherwise
        """
        logger.info("Verifying bundle integrity...")

        # Check directory structure
        required_dirs = [self.models_dir, self.metadata_dir, self.deploy_dir]
        for dir_path in required_dirs:
            if not dir_path.exists():
                logger.error(f"Missing directory: {dir_path}")
                return False

        # Check manifest exists
        manifest_path = self.metadata_dir / "manifest.json"
        if not manifest_path.exists():
            logger.error("Missing manifest.json")
            return False

        # Load and verify manifest
        try:
            with open(manifest_path) as f:
                manifest_data = json.load(f)

            manifest = BundleManifest.from_dict(manifest_data)

            # Verify each artifact
            for artifact in manifest.artifacts:
                # Resolve relative path
                artifact_path = self.bundle_dir / artifact.uri.lstrip("./")

                if not artifact_path.exists():
                    logger.error(f"Missing artifact: {artifact_path}")
                    return False

                # Verify hash
                actual_hash = self._calculate_sha256(artifact_path)
                if actual_hash != artifact.sha256:
                    logger.error(f"Hash mismatch for {artifact_path}")
                    logger.error(f"Expected: {artifact.sha256}")
                    logger.error(f"Actual: {actual_hash}")
                    return False

                # Verify size
                actual_size = artifact_path.stat().st_size
                if actual_size != artifact.size:
                    logger.error(f"Size mismatch for {artifact_path}")
                    return False

            logger.info("✅ Bundle verification passed")
            return True

        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return False

    def get_info(self) -> Dict[str, Any]:
        """
        Get bundle information

        Returns:
            Dictionary with bundle details
        """
        manifest_path = self.metadata_dir / "manifest.json"

        if not manifest_path.exists():
            return {"error": "Bundle not initialized or manifest missing"}

        with open(manifest_path) as f:
            manifest_data = json.load(f)

        # Calculate total size
        total_size = sum(a["size"] for a in manifest_data.get("artifacts", []))

        return {
            "name": self.name,
            "version": self.version,
            "location": str(self.bundle_dir),
            "artifacts": len(manifest_data.get("artifacts", [])),
            "total_size_mb": total_size / (1024 * 1024),
            "manifest": manifest_data,
        }
