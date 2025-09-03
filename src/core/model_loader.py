from pathlib import Path
import logging
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer

logger = logging.getLogger(__name__)


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

        return {
            "model_id": model_id,
            "local_path": Path(local_path),
            "config": config,
            "tokenizer": tokenizer,
            "architecture": architecture,
        }
