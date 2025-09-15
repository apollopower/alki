#!/usr/bin/env python3
"""
Test Bundle Creation Script

Demonstrates how to create a deployment bundle from a GGUF model.
This script can be used for testing and as an example of the bundle API.
"""

import sys
import logging
from pathlib import Path
import tempfile
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.bundle import Bundle, BundleArtifact
from src.core.manifest import ManifestGenerator, ModelInfo
from src.core.validator import GGUFValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_bundle_from_gguf(
    model_path: str,
    output_dir: str,
    bundle_name: str = None,
    quantization: str = "Q4_K_M",
    validate: bool = True
) -> bool:
    """
    Create a deployment bundle from a GGUF model file.
    
    Args:
        model_path: Path to GGUF model file
        output_dir: Output directory for bundle
        bundle_name: Name for the bundle (derived from model if not provided)
        quantization: Quantization profile name
        validate: Whether to validate the model before bundling
        
    Returns:
        True if successful, False otherwise
    """
    model_path = Path(model_path)
    output_dir = Path(output_dir)
    
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        return False
    
    # Derive bundle name from model filename if not provided
    if not bundle_name:
        bundle_name = model_path.stem.lower().replace("_", "-").replace(".", "-")
    
    logger.info(f"Creating bundle '{bundle_name}' from {model_path.name}")
    
    if validate:
        logger.info("Validating GGUF model...")
        validator = GGUFValidator()
        result = validator.validate_file(str(model_path))
        
        if not result.passed:
            logger.error(f"Model validation failed: {result.error}")
            return False
        
        logger.info(f"✅ Model validated successfully")
        logger.info(f"  Context length: {result.context_length}")
        logger.info(f"  Vocab size: {result.vocab_size}")
        logger.info(f"  Embedding size: {result.embedding_size}")
    
    generator = ManifestGenerator()
    capabilities = generator.extract_model_capabilities(model_path)
    
    if capabilities:
        logger.info(f"Extracted model capabilities:")
        logger.info(f"  Context: {capabilities['context_length']}")
        logger.info(f"  Vocab: {capabilities['vocab_size']}")
    else:
        logger.warning("Could not extract model capabilities. Using defaults.")
        capabilities = {"context_length": 2048}  # Safe default
    
    logger.info(f"Creating bundle structure...")
    bundle = Bundle(output_dir, bundle_name)
    bundle.create_structure()
    
    logger.info(f"Adding model to bundle...")
    artifact = bundle.add_model(model_path, quantization)
    
    model_info = ModelInfo(
        architecture="GGUF",
        context_length=capabilities.get("context_length"),
        vocab_size=capabilities.get("vocab_size"),
        embedding_size=capabilities.get("embedding_size"),
        license="Check original model"
    )
    
    # Detect chat template
    chat_template = generator.detect_chat_template(bundle_name)
    logger.info(f"Detected chat template: {chat_template}")
    
    # Create manifests
    logger.info("Generating manifests...")
    
    # Model manifest
    bundle.create_manifest(
        artifacts=[artifact],
        template=chat_template,
        license=model_info.license,
        source_model=model_path.name,
        context_size=capabilities.get("context_length", 2048)
    )
    
    bundle.create_runtime_manifest()
    
    bundle.create_sbom()
    
    bundle.add_readme(bundle_name, [quantization])
    
    bundle.add_license(
        "Please add the appropriate license for your model.\n"
        "Check the original model repository for license information."
    )
    
    logger.info("Creating deployment configurations...")
    
    model_filename = f"{bundle_name}-{quantization.lower()}.gguf"
    
    systemd_config = generator.create_deployment_placeholder(
        "systemd",
        bundle_name,
        model_filename,
        capabilities.get("context_length"),
        chat_template
    )
    (bundle.deploy_dir / "systemd" / f"alki-{bundle_name}.service").write_text(systemd_config)
    
    docker_config = generator.create_deployment_placeholder(
        "docker",
        bundle_name,
        model_filename,
        capabilities.get("context_length"),
        chat_template
    )
    (bundle.deploy_dir / "docker" / "Dockerfile").write_text(docker_config)
    
    logger.info("Verifying bundle integrity...")
    if bundle.verify_bundle():
        logger.info("✅ Bundle created successfully!")
        
        # Print bundle info
        info = bundle.get_info()
        logger.info(f"\nBundle Information:")
        logger.info(f"  Name: {info['name']}")
        logger.info(f"  Version: {info['version']}")
        logger.info(f"  Location: {info['location']}")
        logger.info(f"  Total size: {info['total_size_mb']:.1f} MB")
        
        logger.info(f"\nTo deploy with llama-server:")
        logger.info(f"  llama-server -m {bundle.models_dir}/{model_filename} \\")
        logger.info(f"    --host 0.0.0.0 --port 8080 --ctx-size {capabilities.get('context_length', 2048)} \\")
        logger.info(f"    --chat-format {chat_template}")
        
        return True
    else:
        logger.error("❌ Bundle verification failed!")
        return False


def test_with_huggingface_model():
    """Test bundle creation with a model from HuggingFace"""
    logger.info("=" * 60)
    logger.info("Testing bundle creation with HuggingFace model")
    logger.info("=" * 60)
    
    # This would download and use a real model
    # For testing, we'll create a mock scenario
    with tempfile.TemporaryDirectory() as tmpdir:
        # In a real scenario, you would:
        # 1. Download a GGUF model from HuggingFace
        # 2. Create a bundle from it
        
        # For this test, create a fake model
        model_path = Path(tmpdir) / "test-model.gguf"
        model_path.write_bytes(b"This is a fake GGUF model for testing")
        
        output_dir = Path(tmpdir) / "bundles"
        
        # Note: This will fail validation since it's not a real GGUF
        # Set validate=False for testing
        success = create_bundle_from_gguf(
            str(model_path),
            str(output_dir),
            bundle_name="test-model",
            validate=False
        )
        
        if success:
            logger.info("✅ Test bundle creation successful!")
            
            bundle_dir = output_dir / "test-model"
            logger.info(f"\nCreated files:")
            for file in bundle_dir.rglob("*"):
                if file.is_file():
                    logger.info(f"  {file.relative_to(bundle_dir)}")
        else:
            logger.error("❌ Test bundle creation failed!")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Create a deployment bundle from a GGUF model"
    )
    parser.add_argument(
        "model",
        nargs="?",
        help="Path to GGUF model file"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="./dist",
        help="Output directory for bundle (default: ./dist)"
    )
    parser.add_argument(
        "--name",
        "-n",
        help="Bundle name (derived from model if not provided)"
    )
    parser.add_argument(
        "--quantization",
        "-q",
        default="Q4_K_M",
        help="Quantization profile (default: Q4_K_M)"
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip model validation"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run test with mock data"
    )
    
    args = parser.parse_args()
    
    if args.test:
        # Run test mode
        test_with_huggingface_model()
    elif args.model:
        success = create_bundle_from_gguf(
            args.model,
            args.output,
            args.name,
            args.quantization,
            validate=not args.no_validate
        )
        sys.exit(0 if success else 1)
    else:
        parser.print_help()
        logger.info("\nExamples:")
        logger.info("  # Create bundle from local GGUF file")
        logger.info("  python scripts/test_bundle.py model.gguf -o ./dist")
        logger.info("")
        logger.info("  # Test with mock data")
        logger.info("  python scripts/test_bundle.py --test")


if __name__ == "__main__":
    main()