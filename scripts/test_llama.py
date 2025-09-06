#!/usr/bin/env python3
"""
Test script for LlamaCppModelLoader with configurable GGUF models.
This script downloads and tests quantized models to ensure they work correctly.

Usage:
    python test_llama.py <repo_id> <filename>
    
Example:
    python test_llama.py "Qwen/Qwen3-0.6B-GGUF" "*Qwen3-0.6B-Q8_0.gguf"
    python test_llama.py "Qwen/Qwen2-0.5B-Instruct-GGUF" "*q8_0.gguf"
"""

import sys
import argparse
import logging
import shutil
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.model_loader import LlamaCppModelLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def cleanup_model_cache(repo_id: str):
    """Clean up the Hugging Face cache for a specific model."""
    try:
        from huggingface_hub import scan_cache_dir
        
        logger.info("\n" + "="*50)
        logger.info("CLEANING UP MODEL CACHE")
        logger.info("="*50)
        
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
            
    except ImportError:
        logger.warning("huggingface_hub not available for cache cleanup")
    except Exception as e:
        logger.error(f"Failed to clean up cache: {e}")


def test_llama_model(repo_id: str, filename: str):
    """Test the LlamaCppModelLoader with the specified GGUF model."""
    
    logger.info("Starting LlamaCppModelLoader test")
    logger.info(f"Repository: {repo_id}")
    logger.info(f"Filename pattern: {filename}")
    
    try:
        # Initialize the loader
        loader = LlamaCppModelLoader()
        
        # Load the model
        logger.info("Loading model...")
        model = loader.prepareFromHuggingFace(
            repo_id=repo_id,
            filename=filename,
            verbose=True
        )
        
        # Inspect model properties
        logger.info("\n" + "="*50)
        logger.info("MODEL INSPECTION RESULTS")
        logger.info("="*50)
        
        logger.info(f"Model context length: {model.n_ctx()}")
        logger.info(f"Model vocabulary size: {model.n_vocab()}")
        logger.info(f"Model embedding size: {model.n_embd()}")
        
        # Test basic inference
        logger.info("\n" + "="*50)
        logger.info("TESTING BASIC INFERENCE")
        logger.info("="*50)
        
        test_prompt = "Hello, how are you?"
        logger.info(f"Test prompt: '{test_prompt}'")
        
        # Generate response with basic parameters
        response = model(
            test_prompt,
            max_tokens=50,
            temperature=0.7,
            top_p=0.9,
            echo=False
        )
        
        logger.info("Response generated successfully!")
        logger.info(f"Generated text: '{response['choices'][0]['text'].strip()}'")
        
        # Additional model metadata if available
        logger.info("\n" + "="*50)
        logger.info("ADDITIONAL MODEL METADATA")
        logger.info("="*50)
        
        if hasattr(model, 'metadata'):
            logger.info(f"Model metadata: {model.metadata}")
        else:
            logger.info("No additional metadata available")
            
        logger.info("\n" + "="*50)
        logger.info("TEST COMPLETED SUCCESSFULLY! ✅")
        logger.info("="*50)
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        logger.error("Stack trace:", exc_info=True)
        return False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test LlamaCppModelLoader with configurable GGUF models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "Qwen/Qwen3-0.6B-GGUF" "*Qwen3-0.6B-Q8_0.gguf"
  %(prog)s "Qwen/Qwen2-0.5B-Instruct-GGUF" "*q8_0.gguf"
        """
    )
    
    parser.add_argument(
        "repo_id",
        help="HuggingFace repository ID (e.g., 'Qwen/Qwen3-0.6B-GGUF')"
    )
    
    parser.add_argument(
        "filename",
        help="Filename pattern for the GGUF file (e.g., '*Qwen3-0.6B-Q8_0.gguf')"
    )
    
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Skip cache cleanup after testing"
    )
    
    return parser.parse_args()


def main():
    """Main function to run the test."""
    args = parse_args()
    
    logger.info("Starting GGUF model test script")
    logger.info(f"Repository: {args.repo_id}")
    logger.info(f"Filename: {args.filename}")
    
    success = test_llama_model(args.repo_id, args.filename)
    
    if success:
        logger.info("All tests passed! 🎉")
        if not args.no_cleanup:
            # Clean up the model cache after successful test
            cleanup_model_cache(args.repo_id)
        else:
            logger.info("Skipping cache cleanup (--no-cleanup flag used)")
        sys.exit(0)
    else:
        logger.error("Tests failed! ❌")
        if not args.no_cleanup:
            # Still try to clean up cache even if test failed
            cleanup_model_cache(args.repo_id)
        sys.exit(1)


if __name__ == "__main__":
    main()