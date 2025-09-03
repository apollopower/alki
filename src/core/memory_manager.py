"""Memory monitoring and management for Alki models.

This module provides utilities to monitor system memory usage and apply
memory optimization strategies during model processing pipelines.
"""

import gc
import logging
import os
import warnings
from contextlib import contextmanager
from typing import Optional, Dict

try:
    import psutil
except ImportError:
    psutil = None

logger = logging.getLogger(__name__)


class MemoryManager:
    """Monitors and manages memory usage during model operations."""

    def __init__(
        self, warn_threshold_pct: float = 80.0, critical_threshold_pct: float = 90.0
    ):
        """
        Initialize memory manager.

        Args:
            warn_threshold_pct: Warning threshold as percentage of total memory
            critical_threshold_pct: Critical threshold as percentage of total memory
        """
        self.warn_threshold_pct = warn_threshold_pct
        self.critical_threshold_pct = critical_threshold_pct

        if psutil is None:
            warnings.warn("psutil not available, memory monitoring will be limited")

    def get_memory_info(self) -> Dict[str, float]:
        """Get current memory usage information.

        Returns:
            Dictionary with memory info in MB
        """
        if psutil is None:
            return {"available_mb": -1, "used_pct": -1, "total_mb": -1}

        memory = psutil.virtual_memory()
        return {
            "total_mb": memory.total / (1024 * 1024),
            "available_mb": memory.available / (1024 * 1024),
            "used_pct": memory.percent,
            "used_mb": memory.used / (1024 * 1024),
        }

    def check_memory_threshold(
        self, required_mb: Optional[float] = None
    ) -> tuple[bool, str]:
        """Check if memory usage is within acceptable limits.

        Args:
            required_mb: Estimated memory requirement in MB (optional)

        Returns:
            (is_safe, message) tuple
        """
        memory_info = self.get_memory_info()

        if memory_info["used_pct"] == -1:
            return True, "Memory monitoring not available"

        used_pct = memory_info["used_pct"]
        available_mb = memory_info["available_mb"]

        if used_pct >= self.critical_threshold_pct:
            return (
                False,
                f"Critical memory usage: {used_pct:.1f}% (available: {available_mb:.1f}MB)",
            )

        if required_mb and available_mb < required_mb:
            return (
                False,
                f"Insufficient memory: need {required_mb:.1f}MB, have {available_mb:.1f}MB",
            )

        if used_pct >= self.warn_threshold_pct:
            logger.warning(
                f"High memory usage: {used_pct:.1f}% (available: {available_mb:.1f}MB)"
            )

        return True, f"Memory OK: {used_pct:.1f}% used, {available_mb:.1f}MB available"

    def force_garbage_collection(self) -> None:
        """Force garbage collection to free up memory."""
        logger.debug("Forcing garbage collection")
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # More aggressive cleanup
        for _ in range(5):  # More passes for thorough cleanup
            gc.collect()

        # Force immediate cleanup of unreferenced objects
        if hasattr(gc, "set_debug"):
            original_flags = gc.get_debug()
            gc.set_debug(0)  # Disable debug to avoid keeping references
            for _ in range(3):
                gc.collect()
            gc.set_debug(original_flags)

    def clear_cuda_cache(self) -> None:
        """Clear CUDA cache if PyTorch is available."""
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("Cleared CUDA cache")
        except ImportError:
            pass

    def set_low_memory_mode(self) -> None:
        """Configure environment for low memory usage."""
        # Limit PyTorch memory allocation aggressively
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

        # Reduce HuggingFace cache and memory usage
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Reduce tokenizer memory
        os.environ["OMP_NUM_THREADS"] = "1"  # Limit OpenMP threads
        os.environ["MKL_NUM_THREADS"] = "1"  # Limit Intel MKL threads

        # Try to set memory mapping off to avoid large allocations
        os.environ["TRANSFORMERS_OFFLINE"] = "0"
        os.environ["HF_HUB_OFFLINE"] = "0"

        logger.info("Configured aggressive low memory environment settings")


@contextmanager
def memory_managed_operation(
    memory_manager: MemoryManager,
    operation_name: str,
    estimated_mb: Optional[float] = None,
    force_cleanup: bool = True,
):
    """Context manager for memory-managed operations.

    Args:
        memory_manager: MemoryManager instance
        operation_name: Name of the operation for logging
        estimated_mb: Estimated memory requirement in MB
        force_cleanup: Whether to force cleanup after operation
    """
    # Pre-operation memory check
    is_safe, message = memory_manager.check_memory_threshold(estimated_mb)
    logger.info(f"Starting {operation_name}: {message}")

    if not is_safe:
        logger.warning(f"Memory warning before {operation_name}: {message}")
        memory_manager.force_garbage_collection()
        memory_manager.clear_cuda_cache()

        # Re-check after cleanup
        is_safe, message = memory_manager.check_memory_threshold(estimated_mb)
        if not is_safe:
            raise RuntimeError(f"Insufficient memory for {operation_name}: {message}")

    try:
        yield
    finally:
        # Post-operation cleanup
        if force_cleanup:
            logger.debug(f"Cleaning up after {operation_name}")
            memory_manager.force_garbage_collection()
            memory_manager.clear_cuda_cache()

        # Post-operation memory check
        _, message = memory_manager.check_memory_threshold()
        logger.info(f"Completed {operation_name}: {message}")


def estimate_model_memory_mb(model_size_mb: float) -> float:
    """Estimate memory requirements for model operations.

    Args:
        model_size_mb: Model size in MB

    Returns:
        Estimated memory requirement in MB (includes overhead)
    """
    # Rule of thumb: ONNX export needs ~3x model size due to:
    # - Original model in memory
    # - Intermediate representations during conversion
    # - Final ONNX model
    return model_size_mb * 3.0


# Default memory manager instance
default_memory_manager = MemoryManager()
