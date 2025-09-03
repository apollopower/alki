"""Tests for memory management functionality."""

import pytest
from unittest.mock import Mock, patch

from src.core.memory_manager import (
    MemoryManager,
    memory_managed_operation,
    estimate_model_memory_mb,
    default_memory_manager,
)


class TestMemoryManager:
    """Test core memory management functionality."""

    def test_memory_manager_basic_operations(self):
        """Test basic memory info retrieval and manager initialization."""
        with patch("src.core.memory_manager.psutil") as mock_psutil:
            # Mock memory info
            mock_memory = Mock()
            mock_memory.total = 16 * 1024 * 1024 * 1024  # 16GB
            mock_memory.available = 8 * 1024 * 1024 * 1024  # 8GB available
            mock_memory.percent = 50.0  # 50% used
            mock_memory.used = 8 * 1024 * 1024 * 1024  # 8GB used
            mock_psutil.virtual_memory.return_value = mock_memory

            manager = MemoryManager()

            # Test memory info retrieval
            info = manager.get_memory_info()
            assert info["total_mb"] == pytest.approx(16384, abs=1)
            assert info["available_mb"] == pytest.approx(8192, abs=1)
            assert info["used_pct"] == 50.0

            # Test thresholds
            manager.warn_threshold_pct = 60.0
            manager.critical_threshold_pct = 90.0

            is_safe, message = manager.check_memory_threshold()
            assert is_safe is True
            assert "Memory OK" in message

    def test_check_memory_threshold_levels(self):
        """Test memory threshold detection for warning and critical levels."""
        with patch("src.core.memory_manager.psutil") as mock_psutil:
            manager = MemoryManager(
                warn_threshold_pct=70.0, critical_threshold_pct=90.0
            )

            # Test warning level (80% used)
            mock_memory = Mock()
            mock_memory.total = 8 * 1024 * 1024 * 1024
            mock_memory.available = 1.6 * 1024 * 1024 * 1024  # 1.6GB available
            mock_memory.percent = 80.0
            mock_memory.used = 6.4 * 1024 * 1024 * 1024
            mock_psutil.virtual_memory.return_value = mock_memory

            is_safe, message = manager.check_memory_threshold()
            assert is_safe is True
            assert "80.0%" in message

            # Test critical level (95% used)
            mock_memory.percent = 95.0
            mock_memory.available = 0.4 * 1024 * 1024 * 1024  # 400MB available
            mock_memory.used = 7.6 * 1024 * 1024 * 1024

            is_safe, message = manager.check_memory_threshold()
            assert is_safe is False
            assert "Critical memory usage" in message
            assert "95.0%" in message

            # Test insufficient memory for specific requirement
            is_safe, message = manager.check_memory_threshold(required_mb=1000)
            assert is_safe is False
            assert (
                "Insufficient memory" in message or "Critical memory usage" in message
            )

    def test_memory_managed_operation_context(self):
        """Test the memory-managed operation context manager."""
        with patch("src.core.memory_manager.psutil") as mock_psutil:
            # Mock sufficient memory
            mock_memory = Mock()
            mock_memory.total = 16 * 1024 * 1024 * 1024
            mock_memory.available = 8 * 1024 * 1024 * 1024
            mock_memory.percent = 50.0
            mock_memory.used = 8 * 1024 * 1024 * 1024
            mock_psutil.virtual_memory.return_value = mock_memory

            manager = MemoryManager()

            # Mock garbage collection to verify it's called
            with patch.object(manager, "force_garbage_collection") as mock_gc:
                operation_completed = False

                with memory_managed_operation(
                    manager, "test operation", estimated_mb=1000, force_cleanup=True
                ):
                    operation_completed = True

                assert operation_completed is True
                # Should be called at least once for cleanup
                assert mock_gc.call_count >= 1


def test_estimate_model_memory_mb():
    """Test model memory estimation utility function."""
    # Test basic estimation (3x model size for ONNX export overhead)
    assert estimate_model_memory_mb(1000.0) == 3000.0
    assert estimate_model_memory_mb(500.0) == 1500.0
    assert estimate_model_memory_mb(2200.0) == 6600.0  # TinyLlama case


def test_default_memory_manager_exists():
    """Test that the default memory manager instance is available."""
    assert default_memory_manager is not None
    assert isinstance(default_memory_manager, MemoryManager)
