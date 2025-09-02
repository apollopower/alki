"""
Bundle Format for Alki Edge Deployment

This module defines the Bundle data structure that packages all necessary
artifacts for edge LLM deployment:
- ONNX model files (original and quantized)
- Tokenizer configuration and vocabulary
- Runtime configuration and metadata
- Deployment manifests

The Bundle format ensures self-contained, portable deployments that can
run on edge devices with minimal dependencies.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import yaml


@dataclass
class BundleMetadata:
    """Metadata about the bundled model and creation process."""

    model_id: str  # Original HuggingFace model identifier
    architecture: str  # Model architecture (e.g., "GPT2LMHeadModel")
    alki_version: str  # Version of Alki used to create this bundle
    created_at: datetime  # Bundle creation timestamp
    target: str  # Target deployment (e.g., "cpu", "openvino")
    preset: str  # Optimization preset used (e.g., "balanced")

    # Model characteristics
    original_size_mb: Optional[int] = None  # Original model size in MB
    quantized_size_mb: Optional[int] = None  # Quantized model size in MB
    compression_ratio: Optional[float] = None  # Size reduction ratio

    # Quantization details
    quantization_method: Optional[str] = None  # e.g., "SmoothQuant W8A8"
    quantization_alpha: Optional[float] = None  # SmoothQuant alpha parameter

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        result = {
            "model_id": self.model_id,
            "architecture": self.architecture,
            "alki_version": self.alki_version,
            "created_at": self.created_at.isoformat(),
            "target": self.target,
            "preset": self.preset,
        }

        # Add optional fields if present
        if self.original_size_mb is not None:
            result["original_size_mb"] = self.original_size_mb
        if self.quantized_size_mb is not None:
            result["quantized_size_mb"] = self.quantized_size_mb
        if self.compression_ratio is not None:
            result["compression_ratio"] = round(self.compression_ratio, 3)
        if self.quantization_method is not None:
            result["quantization_method"] = self.quantization_method
        if self.quantization_alpha is not None:
            result["quantization_alpha"] = self.quantization_alpha

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BundleMetadata":
        """Create from dictionary loaded from YAML."""
        return cls(
            model_id=data["model_id"],
            architecture=data["architecture"],
            alki_version=data["alki_version"],
            created_at=datetime.fromisoformat(data["created_at"]),
            target=data["target"],
            preset=data["preset"],
            original_size_mb=data.get("original_size_mb"),
            quantized_size_mb=data.get("quantized_size_mb"),
            compression_ratio=data.get("compression_ratio"),
            quantization_method=data.get("quantization_method"),
            quantization_alpha=data.get("quantization_alpha"),
        )


@dataclass
class RuntimeConfig:
    """Configuration for running the bundled model."""

    # ONNX Runtime configuration
    provider: str = "CPUExecutionProvider"  # Execution provider
    opset_version: int = 14  # ONNX opset version
    use_cache: bool = False  # Whether model uses KV cache

    # Input specifications
    max_sequence_length: int = 512  # Maximum input sequence length
    input_names: List[str] = field(
        default_factory=lambda: ["input_ids", "attention_mask"]
    )
    output_names: List[str] = field(default_factory=lambda: ["logits"])

    # Quantization settings (for runtime reference)
    is_quantized: bool = False
    quantization_format: Optional[str] = None  # e.g., "QDQ", "IntegerOps"
    activation_type: Optional[str] = None  # e.g., "QInt8", "QUInt8"
    weight_type: Optional[str] = None  # e.g., "QInt8"

    # Performance hints
    intra_op_num_threads: Optional[int] = None  # Number of threads for ops
    inter_op_num_threads: Optional[int] = (
        None  # Number of threads for parallel execution
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        result = {
            "provider": self.provider,
            "opset_version": self.opset_version,
            "use_cache": self.use_cache,
            "max_sequence_length": self.max_sequence_length,
            "input_names": self.input_names,
            "output_names": self.output_names,
            "is_quantized": self.is_quantized,
        }

        # Add optional fields
        if self.quantization_format:
            result["quantization_format"] = self.quantization_format
        if self.activation_type:
            result["activation_type"] = self.activation_type
        if self.weight_type:
            result["weight_type"] = self.weight_type
        if self.intra_op_num_threads:
            result["intra_op_num_threads"] = self.intra_op_num_threads
        if self.inter_op_num_threads:
            result["inter_op_num_threads"] = self.inter_op_num_threads

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RuntimeConfig":
        """Create from dictionary loaded from YAML."""
        return cls(
            provider=data.get("provider", "CPUExecutionProvider"),
            opset_version=data.get("opset_version", 14),
            use_cache=data.get("use_cache", False),
            max_sequence_length=data.get("max_sequence_length", 512),
            input_names=data.get("input_names", ["input_ids", "attention_mask"]),
            output_names=data.get("output_names", ["logits"]),
            is_quantized=data.get("is_quantized", False),
            quantization_format=data.get("quantization_format"),
            activation_type=data.get("activation_type"),
            weight_type=data.get("weight_type"),
            intra_op_num_threads=data.get("intra_op_num_threads"),
            inter_op_num_threads=data.get("inter_op_num_threads"),
        )


@dataclass
class BundleArtifacts:
    """Paths to all artifacts within the bundle directory."""

    # Model files
    model_onnx: str = "model.onnx"  # Main ONNX model file
    model_original: Optional[str] = None  # Original unquantized model (if different)

    # Tokenizer files
    tokenizer_dir: str = "tokenizer"
    tokenizer_config: str = "tokenizer/tokenizer_config.json"
    tokenizer_json: Optional[str] = "tokenizer/tokenizer.json"  # Fast tokenizer
    vocab_file: Optional[str] = None  # Traditional vocab (if used)
    merges_file: Optional[str] = None  # BPE merges (if used)
    special_tokens_map: Optional[str] = "tokenizer/special_tokens_map.json"

    # Optional runtime files
    runners_dir: Optional[str] = "runners"  # Directory for runtime launchers

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        result = {
            "model_onnx": self.model_onnx,
            "tokenizer_dir": self.tokenizer_dir,
            "tokenizer_config": self.tokenizer_config,
        }

        # Add optional fields
        if self.model_original:
            result["model_original"] = self.model_original
        if self.tokenizer_json:
            result["tokenizer_json"] = self.tokenizer_json
        if self.vocab_file:
            result["vocab_file"] = self.vocab_file
        if self.merges_file:
            result["merges_file"] = self.merges_file
        if self.special_tokens_map:
            result["special_tokens_map"] = self.special_tokens_map
        if self.runners_dir:
            result["runners_dir"] = self.runners_dir

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BundleArtifacts":
        """Create from dictionary loaded from YAML."""
        return cls(
            model_onnx=data.get("model_onnx", "model.onnx"),
            model_original=data.get("model_original"),
            tokenizer_dir=data.get("tokenizer_dir", "tokenizer"),
            tokenizer_config=data.get(
                "tokenizer_config", "tokenizer/tokenizer_config.json"
            ),
            tokenizer_json=data.get("tokenizer_json", "tokenizer/tokenizer.json"),
            vocab_file=data.get("vocab_file"),
            merges_file=data.get("merges_file"),
            special_tokens_map=data.get(
                "special_tokens_map", "tokenizer/special_tokens_map.json"
            ),
            runners_dir=data.get("runners_dir", "runners"),
        )


@dataclass
class Bundle:
    """
    Complete bundle representation containing all metadata, configuration,
    and artifact references for a deployable LLM.

    This is the main data structure used throughout Alki for managing
    packaged models ready for edge deployment.
    """

    metadata: BundleMetadata
    runtime_config: RuntimeConfig
    artifacts: BundleArtifacts

    # Bundle directory path (set when loaded from disk)
    bundle_path: Optional[Path] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire bundle to dictionary for YAML serialization."""
        return {
            "metadata": self.metadata.to_dict(),
            "runtime": self.runtime_config.to_dict(),
            "artifacts": self.artifacts.to_dict(),
        }

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any], bundle_path: Optional[Path] = None
    ) -> "Bundle":
        """Create Bundle from dictionary loaded from YAML."""
        return cls(
            metadata=BundleMetadata.from_dict(data["metadata"]),
            runtime_config=RuntimeConfig.from_dict(data["runtime"]),
            artifacts=BundleArtifacts.from_dict(data["artifacts"]),
            bundle_path=bundle_path,
        )

    def save_yaml(self, yaml_path: Path) -> None:
        """Save bundle configuration to YAML file."""
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        with open(yaml_path, "w") as f:
            yaml.safe_dump(
                self.to_dict(), f, default_flow_style=False, indent=2, sort_keys=False
            )

    @classmethod
    def load_yaml(cls, yaml_path: Path) -> "Bundle":
        """Load bundle configuration from YAML file."""
        if not yaml_path.exists():
            raise FileNotFoundError(f"Bundle configuration not found: {yaml_path}")

        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        # Set bundle path to the directory containing the YAML file
        bundle_path = yaml_path.parent
        return cls.from_dict(data, bundle_path)

    def get_model_path(self) -> Path:
        """Get absolute path to the main ONNX model file."""
        if self.bundle_path is None:
            raise ValueError("Bundle path not set - load bundle from disk first")
        return self.bundle_path / self.artifacts.model_onnx

    def get_tokenizer_path(self) -> Path:
        """Get absolute path to the tokenizer directory."""
        if self.bundle_path is None:
            raise ValueError("Bundle path not set - load bundle from disk first")
        return self.bundle_path / self.artifacts.tokenizer_dir

    def validate(self) -> List[str]:
        """
        Validate bundle integrity and return list of issues found.

        Returns:
            List of validation error messages. Empty list means bundle is valid.
        """
        issues = []

        if self.bundle_path is None:
            issues.append("Bundle path not set - cannot validate file existence")
            return issues

        # Check that main model file exists
        model_path = self.get_model_path()
        if not model_path.exists():
            issues.append(f"Main model file missing: {self.artifacts.model_onnx}")

        # Check tokenizer directory
        tokenizer_path = self.get_tokenizer_path()
        if not tokenizer_path.exists():
            issues.append(
                f"Tokenizer directory missing: {self.artifacts.tokenizer_dir}"
            )

        # Check required tokenizer files
        if tokenizer_path.exists():
            config_path = self.bundle_path / self.artifacts.tokenizer_config
            if not config_path.exists():
                issues.append(
                    f"Tokenizer config missing: {self.artifacts.tokenizer_config}"
                )

        # Validate metadata consistency
        if self.runtime_config.is_quantized and not self.metadata.quantization_method:
            issues.append(
                "Runtime indicates quantized model but no quantization method in metadata"
            )

        return issues

    def is_valid(self) -> bool:
        """Check if bundle passes validation."""
        return len(self.validate()) == 0


# Helper function for creating bundle directories
def create_bundle_directory_structure(bundle_path: Path) -> None:
    """Create standard bundle directory structure."""
    bundle_path.mkdir(parents=True, exist_ok=True)
    (bundle_path / "tokenizer").mkdir(exist_ok=True)
    (bundle_path / "runners").mkdir(exist_ok=True)
