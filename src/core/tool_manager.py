"""
External tool management for Alki.

This module manages external tools and scripts needed for model conversion,
including downloading, caching, and version management.
"""

import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional
import urllib.request

logger = logging.getLogger(__name__)


class DependencyError(Exception):
    """Raised when required dependencies are missing."""

    pass


class ToolDownloadError(Exception):
    """Raised when tool download fails."""

    pass


class ConversionError(Exception):
    """Raised when model conversion fails."""

    pass


class ToolManager:
    """
    Manages external tools needed for model conversion.

    Handles downloading, caching, and version management of conversion scripts
    and other external dependencies.
    """

    # Pinned version of llama.cpp for stability
    TOOL_VERSION = "b4481"  # Specific commit hash for reproducibility

    # Configuration for conversion script
    # TODO: Add proper SHA256 verification with hash caching for security
    CONVERT_SCRIPT_CONFIG = {
        "name": "convert_hf_to_gguf.py",
        "url": f"https://raw.githubusercontent.com/ggml-org/llama.cpp/{TOOL_VERSION}/convert_hf_to_gguf.py",
    }

    def __init__(self, tools_dir: Optional[Path] = None):
        """
        Initialize ToolManager.

        Args:
            tools_dir: Directory to cache tools (defaults to ~/.alki/tools/)
        """
        if tools_dir is None:
            tools_dir = Path.home() / ".alki" / "tools"

        self.tools_dir = tools_dir
        self.tools_dir.mkdir(parents=True, exist_ok=True)

        logger.debug(f"ToolManager initialized with tools directory: {self.tools_dir}")

    def ensure_conversion_tool(self) -> Path:
        """
        Ensure the HuggingFace to GGUF conversion script is available.

        Downloads and caches the script if not already present.

        Returns:
            Path to the conversion script

        Raises:
            ToolDownloadError: If download fails
        """
        script_name = f"convert_hf_to_gguf_{self.TOOL_VERSION}.py"
        script_path = self.tools_dir / script_name

        if script_path.exists():
            logger.debug(f"Conversion tool already cached: {script_path}")
            return script_path

        logger.info(f"Downloading conversion tool version {self.TOOL_VERSION}...")

        return self._download_tool_with_retry(script_path, max_retries=3)

    def _download_tool_with_retry(
        self, script_path: Path, max_retries: int = 3
    ) -> Path:
        """Download tool with retry logic and proper cleanup."""

        for attempt in range(max_retries):
            try:
                return self._download_tool(script_path)
            except Exception as e:
                logger.warning(f"Download attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise ToolDownloadError(
                        f"Failed to download after {max_retries} attempts: {e}"
                    )

        raise ToolDownloadError("Unexpected error in download retry logic")

    def _download_tool(self, script_path: Path) -> Path:
        """Download tool with proper cleanup using context manager."""

        with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            # Download file
            urllib.request.urlretrieve(self.CONVERT_SCRIPT_CONFIG["url"], tmp_path)

            # Verify download
            self._verify_downloaded_script(tmp_path)

            # Move to final location atomically
            tmp_path.rename(script_path)
            logger.info(f"Successfully downloaded conversion tool to: {script_path}")

            return script_path

        except Exception:
            # Ensure cleanup on any error
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass
            raise

    def _verify_downloaded_script(self, script_path: Path) -> None:
        """Verify the downloaded script is valid."""

        # Check file size
        if script_path.stat().st_size == 0:
            raise ToolDownloadError("Downloaded file is empty")

        # Basic Python script validation
        try:
            with open(script_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Check it contains expected Python patterns
            if not any(
                marker in content
                for marker in [
                    "#!/usr/bin/env python",
                    "import ",
                    "def ",
                    "if __name__",
                ]
            ):
                raise ToolDownloadError(
                    "Downloaded file does not appear to be a Python script"
                )

            # Check for conversion-specific patterns
            if "convert" not in content.lower() or "gguf" not in content.lower():
                raise ToolDownloadError(
                    "Downloaded script does not appear to be a GGUF converter"
                )

        except UnicodeDecodeError:
            raise ToolDownloadError("Downloaded file is not valid UTF-8 text")

    def check_dependencies(self) -> List[str]:
        """
        Check if required Python packages are installed.

        Returns:
            List of missing package names (empty if all are available)
        """
        required_packages = ["torch", "sentencepiece", "gguf", "transformers", "numpy"]

        missing = []

        for package in required_packages:
            try:
                __import__(package)
                logger.debug(f"Package {package} is available")
            except ImportError:
                logger.debug(f"Package {package} is missing")
                missing.append(package)

        return missing

    def check_python_environment(self) -> bool:
        """
        Check if the current Python environment is suitable for conversion.

        Returns:
            True if environment is suitable
        """
        # Check Python version (3.8+ required)
        if sys.version_info < (3, 8):
            logger.warning(
                f"Python {sys.version} may not be compatible (3.8+ recommended)"
            )
            return False

        return True

    def run_conversion_script(
        self,
        script_path: Path,
        hf_model_path: Path,
        output_file: Path,
        timeout_minutes: int = 60,
        **kwargs,
    ) -> bool:
        """
        Run the conversion script with specified parameters.

        Args:
            script_path: Path to conversion script
            hf_model_path: Path to HuggingFace model directory
            output_file: Path for output GGUF file
            timeout_minutes: Timeout in minutes (default 60)
            **kwargs: Additional arguments for the conversion script

        Returns:
            True if conversion succeeded

        Raises:
            ConversionError: If conversion fails
        """
        logger.info(f"Running conversion: {hf_model_path} -> {output_file}")

        # Build command
        cmd = [
            sys.executable,
            str(script_path),
            str(hf_model_path),
            "--outfile",
            str(output_file),
            "--outtype",
            kwargs.get("outtype", "f16"),
        ]

        # Add additional arguments
        if kwargs.get("vocab_only"):
            cmd.append("--vocab-only")

        if kwargs.get("use_temp_file"):
            cmd.append("--use-temp-file")

        logger.debug(f"Conversion command: {' '.join(cmd)}")

        try:
            # Run conversion with configurable timeout
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout_minutes * 60,
            )

            logger.debug("Conversion completed successfully")

            if result.stderr:
                logger.debug(f"Conversion warnings: {result.stderr}")

            # Verify output file was created and is not empty
            if not output_file.exists():
                raise ConversionError("Conversion completed but output file not found")

            if output_file.stat().st_size == 0:
                raise ConversionError("Conversion produced empty output file")

            logger.info(
                f"Conversion successful: {output_file} ({output_file.stat().st_size:,} bytes)"
            )
            return True

        except subprocess.TimeoutExpired:
            logger.error(f"Conversion timed out after {timeout_minutes} minutes")
            raise ConversionError(
                f"Conversion timed out after {timeout_minutes} minutes"
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Conversion failed with return code {e.returncode}")
            logger.error(f"Stderr: {e.stderr}")

            # Extract useful error info from stderr
            error_msg = e.stderr if e.stderr else "Unknown conversion error"
            raise ConversionError(f"Conversion failed: {error_msg}")

    def get_cache_info(self) -> dict:
        """
        Get information about cached tools.

        Returns:
            Dictionary with cache statistics and tool info
        """
        info = {
            "tools_dir": str(self.tools_dir),
            "tools_dir_exists": self.tools_dir.exists(),
            "cached_tools": [],
        }

        if self.tools_dir.exists():
            for tool_file in self.tools_dir.iterdir():
                if tool_file.is_file():
                    info["cached_tools"].append(
                        {
                            "name": tool_file.name,
                            "size": tool_file.stat().st_size,
                            "modified": tool_file.stat().st_mtime,
                        }
                    )

        return info

    def clear_cache(self) -> None:
        """Clear all cached tools."""
        if self.tools_dir.exists():
            import shutil

            shutil.rmtree(self.tools_dir)
            self.tools_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Tool cache cleared")
