from pathlib import Path
import logging
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer
from llama_cpp import Llama

logger = logging.getLogger(__name__)


class LlamaCppModelLoader:
    """Downloads and prepares models for conversion using llama.cpp"""

    def prepareFromHuggingFace(
        self, repo_id: str, filename: str, verbose: bool, n_ctx: int = 512
    ) -> Llama:
        """
        Download model and prepare metadata
        """
        logger.info(f"Downloading: {repo_id}")

        model = Llama.from_pretrained(
            repo_id=repo_id,
            filename=filename,
            verbose=verbose,
            n_ctx=n_ctx,
        )

        logger.info(f"Loaded Llama model with {model.n_ctx} context length")
        return model


class HuggingFaceModelLoader:
    """Downloads and prepares HuggingFace models for conversion."""

    def prepare(self, model_id: str) -> dict:
        """
        Download model and prepare metadata for conversion pipeline.

        Args:
            model_id: HF model ID (e.g., "meta-llama/Llama-3.2-3B-Instruct")

        Returns:
            Dict with model artifacts and metadata
        """
        logger.info(f"Downloading: {model_id}")

        local_path = snapshot_download(repo_id=model_id)
        config = AutoConfig.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        logger.info(
            f"Loaded {config.architectures[0] if config.architectures else 'unknown'} model"
        )

        architecture = config.architectures[0] if config.architectures else "unknown"

        # Estimate model size in MB for memory planning
        size_mb = self._estimate_model_size_mb(config)

        return {
            "model_id": model_id,
            "local_path": Path(local_path),
            "config": config,
            "tokenizer": tokenizer,
            "architecture": architecture,
            "size_mb": size_mb,
        }

    def _estimate_model_size_mb(self, config) -> float:
        """Estimate model size in MB based on configuration parameters."""
        try:
            # Basic estimation based on common model parameters
            # This is a rough estimate for memory planning purposes

            if hasattr(config, "num_parameters"):
                # If the config includes parameter count
                return (
                    config.num_parameters * 4 / (1024 * 1024)
                )  # 4 bytes per FP32 parameter

            # Fallback estimation based on architecture-specific patterns
            hidden_size = getattr(config, "hidden_size", 768)
            vocab_size = getattr(config, "vocab_size", 50000)
            num_layers = getattr(
                config, "num_hidden_layers", getattr(config, "num_layers", 12)
            )
            intermediate_size = getattr(config, "intermediate_size", hidden_size * 4)

            # Rough parameter count estimation
            # Embedding layer: vocab_size * hidden_size
            # Each transformer layer: ~4 * hidden_size^2 + intermediate weights
            # Final layer norm and LM head

            embedding_params = vocab_size * hidden_size
            layer_params = num_layers * (
                4 * hidden_size * hidden_size + 2 * hidden_size * intermediate_size
            )
            head_params = vocab_size * hidden_size

            total_params = embedding_params + layer_params + head_params
            size_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per FP32 parameter

            logger.debug(
                f"Estimated model size: {size_mb:.1f}MB ({total_params:,} parameters)"
            )
            return size_mb

        except Exception as e:
            logger.warning(f"Failed to estimate model size: {e}")
            return 1000.0  # Default fallback
