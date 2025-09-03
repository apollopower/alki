"""
ONNX GenAI Runtime for Alki Edge Deployment

This module provides runtime inference capabilities for deployed bundles
using ONNX Runtime GenAI. It handles model loading, tokenization, and
text generation for edge deployment scenarios.
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

from .bundle import Bundle
from .constants import ExecutionProviders

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for text generation parameters."""

    max_tokens: int = 100
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True

    def to_search_options(self) -> Dict[str, Any]:
        """Convert to ONNX GenAI search options format."""
        return {
            "max_length": self.max_tokens,
            "do_sample": self.do_sample,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "temperature": self.temperature,
        }


class OnnxGenAIRunner:
    """
    Runtime inference engine for ONNX GenAI models.

    Provides text generation capabilities for deployed bundles using
    ONNX Runtime GenAI backend. Handles model loading, tokenization,
    and generation with configurable sampling parameters.
    """

    def __init__(self, bundle: Bundle):
        """
        Initialize runtime with bundle configuration.

        Args:
            bundle: Loaded bundle containing model and configuration
        """
        self.bundle = bundle
        self.model = None
        self.tokenizer = None
        self._model_loaded = False

        logger.info(f"Initialized runtime for bundle: {bundle.metadata.model_id}")

    def load_model(self) -> None:
        """
        Load ONNX model and tokenizer from bundle.

        Raises:
            RuntimeError: If model loading fails
            ImportError: If onnxruntime-genai is not available
        """
        try:
            import onnxruntime_genai as og
        except ImportError as e:
            raise ImportError(
                "onnxruntime-genai is required for runtime inference. "
                "Install with: pip install onnxruntime-genai"
            ) from e

        if not self.bundle.bundle_path:
            raise RuntimeError("Bundle path not set - cannot load model")

        bundle_path = self.bundle.bundle_path
        model_path = bundle_path / self.bundle.artifacts.model_onnx

        if not model_path.exists():
            raise RuntimeError(f"Model file not found: {model_path}")

        logger.info(f"Loading model from: {model_path.parent}")

        try:
            # Load model - ONNX GenAI expects the directory containing the model
            self.model = og.Model(str(model_path.parent))
            self.tokenizer = og.Tokenizer(self.model)

            self._model_loaded = True
            logger.info("✓ Model and tokenizer loaded successfully")

        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}") from e

    def generate(self, prompt: str, config: Optional[GenerationConfig] = None) -> str:
        """
        Generate text completion for the given prompt.

        Args:
            prompt: Input text prompt
            config: Generation configuration (uses defaults if None)

        Returns:
            Generated text completion

        Raises:
            RuntimeError: If model is not loaded or generation fails
        """
        if not self._model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        config = config or GenerationConfig()

        logger.info(
            f"Generating with max_tokens={config.max_tokens}, "
            f"temperature={config.temperature}, top_p={config.top_p}"
        )

        try:
            import onnxruntime_genai as og

            # Encode input prompt
            input_tokens = self.tokenizer.encode(prompt)
            logger.debug(f"Encoded prompt to {len(input_tokens)} tokens")

            # Create generation parameters
            params = og.GeneratorParams(self.model)
            params.input_ids = input_tokens

            # Set search options
            search_options = config.to_search_options()
            params.set_search_options(**search_options)

            # Generate output tokens
            output_tokens = self.model.generate(params)[0]

            # Decode generated text
            generated_text = self.tokenizer.decode(output_tokens)

            logger.info(f"Generated {len(output_tokens)} tokens")
            return generated_text

        except Exception as e:
            raise RuntimeError(f"Text generation failed: {str(e)}") from e

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        if not self._model_loaded:
            return {"status": "not_loaded"}

        metadata = self.bundle.metadata
        runtime = self.bundle.runtime_config

        return {
            "status": "loaded",
            "model_id": metadata.model_id,
            "architecture": metadata.architecture,
            "target": metadata.target,
            "provider": runtime.provider,
            "is_quantized": runtime.is_quantized,
            "quantization_method": metadata.quantization_method,
            "bundle_path": (
                str(self.bundle.bundle_path) if self.bundle.bundle_path else None
            ),
        }

    def validate_compatibility(self) -> Dict[str, Any]:
        """
        Validate runtime compatibility with current environment.

        Returns:
            Dictionary with compatibility information
        """
        compatibility = {
            "compatible": True,
            "issues": [],
            "warnings": [],
        }

        # Check if bundle requires specific execution provider
        runtime = self.bundle.runtime_config
        if runtime.provider != ExecutionProviders.CPU:
            compatibility["warnings"].append(
                f"Bundle configured for {runtime.provider} but runtime uses CPU"
            )

        # Check if onnxruntime-genai is available
        try:
            import onnxruntime_genai

            logger.debug(f"ONNX GenAI version: {onnxruntime_genai.__version__}")
        except ImportError:
            compatibility["compatible"] = False
            compatibility["issues"].append("onnxruntime-genai not available")

        return compatibility
