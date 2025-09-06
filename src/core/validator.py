"""
GGUF Model Validator

Validates GGUF models to ensure they load correctly and can perform basic inference.
This provides quality assurance for GGUF models before deployment.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import time
import shutil

from .model_loader import LlamaCppModelLoader

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Results from GGUF model validation"""

    model_path: str
    passed: bool
    load_time_ms: float
    inference_time_ms: float
    context_length: Optional[int] = None
    vocab_size: Optional[int] = None
    embedding_size: Optional[int] = None
    error: Optional[str] = None
    inference_output: Optional[str] = None
    repo_id: Optional[str] = None  # For cleanup purposes

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


class GGUFValidator:
    """Validates GGUF models for correctness and basic functionality"""

    def __init__(
        self, test_prompt: str = "Here is a short history of Alki beach in Seattle."
    ):
        """
        Initialize validator

        Args:
            test_prompt: Prompt to use for inference testing
        """
        self.test_prompt = test_prompt
        self.loader = LlamaCppModelLoader()

    def validate_file(
        self, model_path: str, max_tokens: int = 20, n_ctx: int = 512
    ) -> ValidationResult:
        """
        Validate a GGUF model file

        Args:
            model_path: Path to GGUF model file
            max_tokens: Maximum tokens for inference test

        Returns:
            ValidationResult with test results
        """
        logger.info(f"Validating GGUF model: {model_path}")

        start_time = time.time()

        try:
            # Check if file exists
            if not Path(model_path).exists():
                return ValidationResult(
                    model_path=model_path,
                    passed=False,
                    load_time_ms=0,
                    inference_time_ms=0,
                    error=f"Model file not found: {model_path}",
                )

            # Test model loading
            logger.debug("Testing model loading...")
            load_start = time.time()

            # Load model directly from file path
            from llama_cpp import Llama

            model = Llama(model_path=model_path, verbose=False, n_ctx=n_ctx)

            load_time = (time.time() - load_start) * 1000
            logger.debug(f"Model loaded in {load_time:.1f}ms")

            # Extract model properties
            context_length = model.n_ctx()
            vocab_size = model.n_vocab()
            embedding_size = model.n_embd()

            logger.debug(
                f"Model properties - ctx: {context_length}, vocab: {vocab_size}, embd: {embedding_size}"
            )

            # Test basic inference
            logger.debug("Testing inference...")
            inference_start = time.time()

            response = model(
                self.test_prompt,
                max_tokens=max_tokens,
                temperature=0.1,  # Low temperature for consistent results
                echo=False,
            )

            inference_time = (time.time() - inference_start) * 1000

            # Extract generated text
            generated_text = response["choices"][0]["text"].strip()
            logger.debug(f"Generated text: '{generated_text}'")

            total_time = (time.time() - start_time) * 1000
            logger.info(f"Validation completed in {total_time:.1f}ms")

            return ValidationResult(
                model_path=model_path,
                passed=True,
                load_time_ms=load_time,
                inference_time_ms=inference_time,
                context_length=context_length,
                vocab_size=vocab_size,
                embedding_size=embedding_size,
                inference_output=generated_text,
            )

        except Exception as e:
            error_msg = f"Validation failed: {str(e)}"
            logger.error(error_msg, exc_info=True)

            return ValidationResult(
                model_path=model_path,
                passed=False,
                load_time_ms=(time.time() - start_time) * 1000,
                inference_time_ms=0,
                error=error_msg,
            )

    def validate_huggingface(
        self, repo_id: str, filename: str, max_tokens: int = 20, n_ctx: int = 512
    ) -> ValidationResult:
        """
        Validate a GGUF model from HuggingFace Hub

        Args:
            repo_id: HuggingFace repository ID
            filename: GGUF filename pattern
            max_tokens: Maximum tokens for inference test

        Returns:
            ValidationResult with test results
        """
        logger.info(f"Validating HuggingFace GGUF model: {repo_id}/{filename}")

        start_time = time.time()

        try:
            # Load model using our existing loader
            logger.debug("Loading model from HuggingFace...")
            load_start = time.time()

            model = self.loader.prepareFromHuggingFace(
                repo_id=repo_id, filename=filename, verbose=False, n_ctx=n_ctx
            )

            load_time = (time.time() - load_start) * 1000
            logger.debug(f"Model loaded in {load_time:.1f}ms")

            # Extract model properties
            context_length = model.n_ctx()
            vocab_size = model.n_vocab()
            embedding_size = model.n_embd()

            logger.debug(
                f"Model properties - ctx: {context_length}, vocab: {vocab_size}, embd: {embedding_size}"
            )

            # Test basic inference
            logger.debug("Testing inference...")
            inference_start = time.time()

            response = model(
                self.test_prompt,
                max_tokens=max_tokens,
                temperature=0.1,  # Low temperature for consistent results
                echo=False,
            )

            inference_time = (time.time() - inference_start) * 1000

            # Extract generated text
            generated_text = response["choices"][0]["text"].strip()
            logger.debug(f"Generated text: '{generated_text}'")

            total_time = (time.time() - start_time) * 1000
            logger.info(f"Validation completed in {total_time:.1f}ms")

            return ValidationResult(
                model_path=f"{repo_id}/{filename}",
                passed=True,
                load_time_ms=load_time,
                inference_time_ms=inference_time,
                context_length=context_length,
                vocab_size=vocab_size,
                embedding_size=embedding_size,
                inference_output=generated_text,
            )

        except Exception as e:
            error_msg = f"Validation failed: {str(e)}"
            logger.error(error_msg, exc_info=True)

            return ValidationResult(
                model_path=f"{repo_id}/{filename}",
                passed=False,
                load_time_ms=(time.time() - start_time) * 1000,
                inference_time_ms=0,
                error=error_msg,
            )

    def print_result(self, result: ValidationResult) -> None:
        """Print validation result in a human-readable format"""
        print("\n" + "=" * 60)
        print("GGUF MODEL VALIDATION RESULTS")
        print("=" * 60)

        print(f"Model: {result.model_path}")
        print(f"Status: {'✅ PASSED' if result.passed else '❌ FAILED'}")

        if result.passed:
            print(f"Load Time: {result.load_time_ms:.1f}ms")
            print(f"Inference Time: {result.inference_time_ms:.1f}ms")
            print(f"Context Length: {result.context_length}")
            print(f"Vocabulary Size: {result.vocab_size:,}")
            print(f"Embedding Size: {result.embedding_size}")

            if result.inference_output:
                print(f"Sample Output: '{result.inference_output}'")
        else:
            print(f"Error: {result.error}")

        print("=" * 60)

    def cleanup_model_cache(self, repo_id: str) -> bool:
        """Clean up the Hugging Face cache for a specific model.

        Args:
            repo_id: HuggingFace repository ID

        Returns:
            True if cleanup was successful, False otherwise
        """
        try:
            from huggingface_hub import scan_cache_dir

            logger.info("\n" + "=" * 50)
            logger.info("CLEANING UP MODEL CACHE")
            logger.info("=" * 50)

            # Get cache info
            cache_info = scan_cache_dir()

            # Find the repo in cache
            repo_found = False
            for repo in cache_info.repos:
                if repo.repo_id == repo_id:
                    repo_found = True
                    repo_size = repo.size_on_disk_str
                    logger.info(f"Found cached model: {repo_id}")
                    logger.info(f"Cache size: {repo_size}")
                    logger.info(f"Cache location: {repo.repo_path}")

                    # Delete the entire repository cache
                    logger.info("Deleting model cache...")
                    shutil.rmtree(repo.repo_path)
                    logger.info("✅ Model cache deleted successfully!")
                    break

            if not repo_found:
                logger.info(f"No cache found for {repo_id}")

            return True

        except ImportError:
            logger.warning("huggingface_hub not available for cache cleanup")
            return False
        except Exception as e:
            logger.error(f"Failed to clean up cache: {e}")
            return False

    def validate_and_cleanup(
        self,
        repo_id: str,
        filename: str,
        max_tokens: int = 20,
        cleanup: bool = True,
        n_ctx: int = 512,
    ) -> ValidationResult:
        """Validate a GGUF model and optionally clean up cache afterward.

        Args:
            repo_id: HuggingFace repository ID
            filename: GGUF filename pattern
            max_tokens: Maximum tokens for inference test
            cleanup: Whether to clean up cache after validation

        Returns:
            ValidationResult with test results
        """
        # Perform validation
        result = self.validate_huggingface(repo_id, filename, max_tokens, n_ctx)

        # Add repo_id to result for cleanup tracking
        result.repo_id = repo_id

        # Clean up cache if requested and validation passed
        if cleanup:
            logger.info("\nCleaning up model cache...")
            cleanup_success = self.cleanup_model_cache(repo_id)
            if cleanup_success:
                logger.info("Cache cleanup completed")
            else:
                logger.warning(
                    "Cache cleanup failed but validation results are still valid"
                )

        return result
