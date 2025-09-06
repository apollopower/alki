#!/usr/bin/env python3
"""
Test script for GGUFValidator to ensure validation logic works correctly.
This script tests both HuggingFace and local file validation capabilities.

Usage:
    python scripts/test_validator.py
    python scripts/test_validator.py --repo-id "Qwen/Qwen3-0.6B-GGUF" --filename "*q8_0.gguf"
"""

import sys
import argparse
import logging
from pathlib import Path
import json

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.validator import GGUFValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_huggingface_validation(
    repo_id: str, filename: str, max_tokens: int = 30, cleanup: bool = True
):
    """Test validation of a HuggingFace GGUF model"""
    logger.info(f"Testing HuggingFace validation: {repo_id}/{filename}")
    logger.info(f"Cleanup after validation: {cleanup}")

    validator = GGUFValidator(
        test_prompt="Here is a brief history of Alki Beach in Seattle."
    )

    result = validator.validate_and_cleanup(
        repo_id=repo_id, filename=filename, max_tokens=max_tokens, cleanup=cleanup
    )

    validator.print_result(result)

    # Also save results to JSON for programmatic use
    results_path = Path("validation_results.json")
    with open(results_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    logger.info(f"Results saved to: {results_path}")

    return result.passed


def test_multiple_models(cleanup: bool = True):
    """Test validation with multiple known GGUF models"""
    test_cases = [
        {
            "name": "Qwen2 0.5B Instruct Q8_0",
            "repo_id": "Qwen/Qwen2-0.5B-Instruct-GGUF",
            "filename": "*q8_0.gguf",
        },
        {
            "name": "Qwen3 0.6B Q8_0",
            "repo_id": "Qwen/Qwen3-0.6B-GGUF",
            "filename": "*q8_0.gguf",
        },
    ]

    logger.info("Running validation tests on multiple models...")
    results = {}

    for test_case in test_cases:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing: {test_case['name']}")
        logger.info(f"{'='*60}")

        try:
            passed = test_huggingface_validation(
                repo_id=test_case["repo_id"],
                filename=test_case["filename"],
                max_tokens=20,
                cleanup=cleanup,
            )
            results[test_case["name"]] = "PASSED" if passed else "FAILED"
        except Exception as e:
            logger.error(f"Test failed with error: {e}")
            results[test_case["name"]] = f"ERROR: {str(e)}"

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("VALIDATION TEST SUMMARY")
    logger.info(f"{'='*60}")

    for test_name, result in results.items():
        status_emoji = "✅" if result == "PASSED" else "❌"
        logger.info(f"{status_emoji} {test_name}: {result}")

    passed_count = sum(1 for r in results.values() if r == "PASSED")
    total_count = len(results)

    logger.info(f"\nOverall: {passed_count}/{total_count} tests passed")

    return passed_count == total_count


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test GGUFValidator functionality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s  # Run tests on multiple models
  %(prog)s --repo-id "Qwen/Qwen3-0.6B-GGUF" --filename "*q8_0.gguf"
  %(prog)s --repo-id "Qwen/Qwen3-0.6B-GGUF" --filename "*Q8_0.gguf" --no-cleanup
        """,
    )

    parser.add_argument("--repo-id", help="HuggingFace repository ID to test")

    parser.add_argument("--filename", help="Filename pattern for the GGUF file")

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=30,
        help="Maximum tokens for inference test (default: 30)",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    parser.add_argument(
        "--no-cleanup", action="store_true", help="Skip cache cleanup after testing"
    )

    return parser.parse_args()


def main():
    """Main function to run validator tests."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("Starting GGUFValidator test script")

    if args.repo_id and args.filename:
        # Single model test
        success = test_huggingface_validation(
            repo_id=args.repo_id,
            filename=args.filename,
            max_tokens=args.max_tokens,
            cleanup=not args.no_cleanup,
        )

        if success:
            logger.info("Single model validation test passed! 🎉")
            sys.exit(0)
        else:
            logger.error("Single model validation test failed! ❌")
            sys.exit(1)
    else:
        # Multiple model tests
        success = test_multiple_models(cleanup=not args.no_cleanup)

        if success:
            logger.info("All validation tests passed! 🎉")
            sys.exit(0)
        else:
            logger.error("Some validation tests failed! ❌")
            sys.exit(1)


if __name__ == "__main__":
    main()
