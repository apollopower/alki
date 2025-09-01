from pathlib import Path
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer


class HuggingFaceModelLoader:
    """Downloads and prepares HF models for conversion - v1"""

    def prepare(self, model_id: str) -> dict:
        """
        Download model and prepare metadata for conversion pipeline.

        Args:
            model_id: HF model ID (e.g., "meta-llama/Llama-3.2-3B-Instruct")

        Returns:
            Dict with model artifacts and metadata
        """
        print(f"Downloading: {model_id}")

        # Download to HF cache (no memory load)
        local_path = snapshot_download(repo_id=model_id)

        # Load just config and tokenizer (lightweight)
        config = AutoConfig.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        return {
            "model_id": model_id,
            "local_path": Path(local_path),
            "config": config,
            "tokenizer": tokenizer,
            "architecture": (
                config.architectures[0] if config.architectures else "unknown"
            ),
        }
