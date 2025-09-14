"""
Model type detection and format analysis.

This module provides utilities to detect what type of model we're working with
and determine the appropriate processing strategy.
"""

import logging
from enum import Enum
from pathlib import Path
from typing import Optional

from huggingface_hub import list_repo_files, repo_exists
from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Supported model types and their sources"""

    LOCAL_GGUF = "local_gguf"
    HF_GGUF = "huggingface_gguf"
    HF_PYTORCH = "huggingface_pytorch"
    UNKNOWN = "unknown"


class ModelDetector:
    """
    Detect model format and source to determine processing strategy.

    This class helps the pack command decide whether to:
    - Use a local GGUF file directly
    - Download a GGUF from HuggingFace
    - Convert a PyTorch model from HuggingFace to GGUF
    """

    def detect_model_type(
        self, model_id: str, filename: Optional[str] = None
    ) -> ModelType:
        """
        Detect the type and format of the specified model.

        Args:
            model_id: Model identifier (local path or HuggingFace repo ID)
            filename: Optional filename pattern for HuggingFace models

        Returns:
            ModelType indicating the detected model format and source
        """
        logger.debug(f"Detecting model type for: {model_id}")

        # Check if it's a local file first
        local_path = Path(model_id)
        if local_path.exists():
            if local_path.is_file() and local_path.suffix == ".gguf":
                logger.debug("Detected local GGUF file")
                return ModelType.LOCAL_GGUF
            else:
                logger.debug("Local path exists but not a GGUF file")
                return ModelType.UNKNOWN

        # Check if it's a HuggingFace repository
        try:
            if not repo_exists(model_id):
                logger.debug(f"Repository {model_id} does not exist")
                return ModelType.UNKNOWN

            # Get repository files
            files = list_repo_files(model_id)
            logger.debug(f"Found {len(files)} files in repository")

            # Check for GGUF files
            gguf_files = [f for f in files if f.endswith(".gguf")]
            if gguf_files:
                logger.debug(f"Found GGUF files: {gguf_files}")
                return ModelType.HF_GGUF

            # Check for PyTorch/SafeTensors model files
            safetensor_files = [f for f in files if f.endswith(".safetensors")]
            bin_files = [f for f in files if f.endswith(".bin")]

            has_config = "config.json" in files
            has_weights = bool(safetensor_files or bin_files)

            if has_config and has_weights:
                logger.debug("Found PyTorch/SafeTensors model files")
                return ModelType.HF_PYTORCH

            logger.debug("Repository exists but no recognized model format found")
            return ModelType.UNKNOWN

        except (RepositoryNotFoundError, RevisionNotFoundError) as e:
            logger.debug(f"Repository access error: {e}")
            return ModelType.UNKNOWN
        except Exception as e:
            logger.warning(f"Error checking repository {model_id}: {e}")
            return ModelType.UNKNOWN
