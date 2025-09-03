"""
Standard ONNX Runtime for Alki Edge Deployment

This module provides runtime inference capabilities using standard ONNX Runtime
instead of ONNX GenAI. This approach is more compatible with our current
export pipeline and doesn't require special model formats.
"""

import logging
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass

from .bundle import Bundle

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for text generation parameters."""

    max_tokens: int = 100
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True


class OnnxRuntimeRunner:
    """
    Standard ONNX Runtime inference engine for deployed bundles.

    Provides text generation capabilities using standard ONNX Runtime
    with tokenizers from transformers library. This approach is more
    compatible with our current export pipeline.
    """

    def __init__(self, bundle: Bundle):
        """
        Initialize runtime with bundle configuration.

        Args:
            bundle: Loaded bundle containing model and configuration
        """
        self.bundle = bundle
        self.session = None
        self.tokenizer = None
        self._model_loaded = False

        logger.info(f"Initialized ONNX runtime for bundle: {bundle.metadata.model_id}")

    def load_model(self) -> None:
        """
        Load ONNX model and tokenizer from bundle.

        Raises:
            RuntimeError: If model loading fails
            ImportError: If required libraries are not available
        """
        try:
            import onnxruntime as ort
            from transformers import AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "onnxruntime and transformers are required for runtime inference. "
                "They should be installed by default."
            ) from e

        if not self.bundle.bundle_path:
            raise RuntimeError("Bundle path not set - cannot load model")

        bundle_path = self.bundle.bundle_path
        model_path = bundle_path / self.bundle.artifacts.model_onnx
        tokenizer_path = bundle_path / self.bundle.artifacts.tokenizer_dir

        if not model_path.exists():
            raise RuntimeError(f"Model file not found: {model_path}")

        if not tokenizer_path.exists():
            raise RuntimeError(f"Tokenizer directory not found: {tokenizer_path}")

        logger.info(f"Loading ONNX model from: {model_path}")
        logger.info(f"Loading tokenizer from: {tokenizer_path}")

        try:
            # Create ONNX Runtime session
            providers = [self.bundle.runtime_config.provider]
            self.session = ort.InferenceSession(str(model_path), providers=providers)

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

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
            # Encode input prompt
            inputs = self.tokenizer(prompt, return_tensors="np", padding=True)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            # Create position_ids
            seq_length = input_ids.shape[1]
            position_ids = np.arange(0, seq_length, dtype=np.int64).reshape(1, -1)

            logger.debug(f"Encoded prompt to shape {input_ids.shape}")

            # Generate tokens one by one
            generated_tokens = []
            current_input_ids = input_ids
            current_attention_mask = attention_mask
            current_position_ids = position_ids

            for _ in range(config.max_tokens):
                # Run inference
                ort_inputs = {
                    "input_ids": current_input_ids,
                    "attention_mask": current_attention_mask,
                    "position_ids": current_position_ids,
                }

                outputs = self.session.run(None, ort_inputs)
                logits = outputs[0]  # Shape: (batch_size, seq_len, vocab_size)

                # Get logits for the last token
                next_token_logits = logits[0, -1, :]  # Shape: (vocab_size,)

                # Apply temperature
                if config.temperature != 1.0:
                    next_token_logits = next_token_logits / config.temperature

                # Apply sampling
                if config.do_sample:
                    # Convert to probabilities
                    probs = self._softmax(next_token_logits)

                    # Apply top-k filtering
                    if config.top_k > 0:
                        indices_to_remove = (
                            next_token_logits
                            < np.partition(next_token_logits, -config.top_k)[
                                -config.top_k
                            ]
                        )
                        next_token_logits[indices_to_remove] = -float("Inf")
                        probs = self._softmax(next_token_logits)

                    # Apply top-p filtering
                    if config.top_p < 1.0:
                        sorted_indices = np.argsort(probs)[::-1]
                        cumulative_probs = np.cumsum(probs[sorted_indices])
                        sorted_indices_to_remove = cumulative_probs > config.top_p
                        sorted_indices_to_remove[1:] = sorted_indices_to_remove[
                            :-1
                        ].copy()
                        sorted_indices_to_remove[0] = False
                        indices_to_remove = sorted_indices[sorted_indices_to_remove]
                        probs[indices_to_remove] = 0
                        probs = probs / np.sum(probs)  # Renormalize

                    # Sample next token
                    next_token = np.random.choice(len(probs), p=probs)
                else:
                    # Greedy decoding
                    next_token = np.argmax(next_token_logits)

                generated_tokens.append(next_token)

                # Check if we hit EOS token
                if next_token == self.tokenizer.eos_token_id:
                    break

                # Update input for next iteration
                next_token_array = np.array([[next_token]], dtype=np.int64)
                current_input_ids = np.concatenate(
                    [current_input_ids, next_token_array], axis=1
                )

                # Update attention mask
                next_attention = np.ones((1, 1), dtype=np.int64)
                current_attention_mask = np.concatenate(
                    [current_attention_mask, next_attention], axis=1
                )

                # Update position_ids
                next_position = np.array(
                    [[current_position_ids.shape[1]]], dtype=np.int64
                )
                current_position_ids = np.concatenate(
                    [current_position_ids, next_position], axis=1
                )

            # Decode generated tokens
            if generated_tokens:
                generated_text = self.tokenizer.decode(
                    generated_tokens, skip_special_tokens=True
                )
            else:
                generated_text = ""

            logger.info(f"Generated {len(generated_tokens)} tokens")
            return generated_text

        except Exception as e:
            raise RuntimeError(f"Text generation failed: {str(e)}") from e

    def _softmax(self, x):
        """Apply softmax function to array."""
        exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
        return exp_x / np.sum(exp_x)

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

        # Check if ONNX Runtime is available
        try:
            import onnxruntime

            logger.debug(f"ONNX Runtime version: {onnxruntime.__version__}")
        except ImportError:
            compatibility["compatible"] = False
            compatibility["issues"].append("onnxruntime not available")

        # Check if transformers is available
        try:
            import transformers

            logger.debug(f"Transformers version: {transformers.__version__}")
        except ImportError:
            compatibility["compatible"] = False
            compatibility["issues"].append("transformers not available")

        return compatibility
