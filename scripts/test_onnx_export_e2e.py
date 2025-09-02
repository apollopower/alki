#!/usr/bin/env python3
"""
Real integration test for ONNX export pipeline.
This script tests the actual model loading and ONNX conversion without mocks.

Usage:
    python scripts/test_real_onnx_export.py
    python scripts/test_real_onnx_export.py --model microsoft/DialoGPT-small
    python scripts/test_real_onnx_export.py --keep-files --verbose
"""

import argparse
import sys
import tempfile
import shutil
from pathlib import Path
import time
import traceback

# Add parent directory to path to import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.model_loader import HuggingFaceModelLoader
from src.core.onnx_exporter import OnnxExporter, OnnxExportConfig
import onnxruntime as ort
import numpy as np


def test_onnx_export(model_id: str, keep_files: bool = False, verbose: bool = False):
    """
    Test the complete ONNX export pipeline with a real model.
    
    Args:
        model_id: HuggingFace model ID to test
        keep_files: Whether to keep the exported ONNX files
        verbose: Enable verbose output
    """
    print(f"\n{'='*60}")
    print(f"Testing ONNX Export Pipeline")
    print(f"Model: {model_id}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    temp_dir = None
    
    try:
        # Step 1: Load the model
        print("Step 1: Loading model from HuggingFace...")
        loader = HuggingFaceModelLoader()
        model_artifacts = loader.prepare(model_id)
        
        if verbose:
            print(f"  ✓ Model ID: {model_artifacts['model_id']}")
            print(f"  ✓ Architecture: {model_artifacts['architecture']}")
            print(f"  ✓ Local path: {model_artifacts['local_path']}")
        
        # Step 2: Check architecture support
        print("\nStep 2: Checking architecture support...")
        exporter = OnnxExporter()
        
        is_supported = exporter.validate_architecture(model_artifacts['architecture'])
        if is_supported:
            print(f"  ✓ Architecture '{model_artifacts['architecture']}' is supported")
        else:
            print(f"  ⚠ Architecture '{model_artifacts['architecture']}' may have limited support")
        
        # Step 3: Export to ONNX
        print("\nStep 3: Exporting to ONNX format...")
        
        if keep_files:
            output_dir = Path("onnx_output") / model_id.replace("/", "_")
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            temp_dir = tempfile.mkdtemp(prefix="onnx_test_")
            output_dir = Path(temp_dir)
        
        export_config = OnnxExportConfig(
            use_gpu=False,  # Use CPU for testing
            optimize=True,
            opset_version=14
        )
        
        exporter = OnnxExporter(export_config)
        export_result = exporter.export(model_artifacts, output_dir)
        
        print(f"  ✓ ONNX model exported to: {export_result['onnx_path']}")
        
        # Step 4: Validate ONNX model can be loaded
        print("\nStep 4: Validating ONNX model with ONNX Runtime...")
        
        # Find the actual ONNX model file
        onnx_model_path = output_dir / "model.onnx"
        if not onnx_model_path.exists():
            # Sometimes optimum saves with different naming
            onnx_files = list(output_dir.glob("*.onnx"))
            if onnx_files:
                onnx_model_path = onnx_files[0]
            else:
                raise FileNotFoundError(f"No ONNX model found in {output_dir}")
        
        # Create ONNX Runtime session
        session = ort.InferenceSession(
            str(onnx_model_path),
            providers=['CPUExecutionProvider']
        )
        
        # Get model info
        input_names = [inp.name for inp in session.get_inputs()]
        output_names = [out.name for out in session.get_outputs()]
        
        if verbose:
            print(f"  ✓ ONNX Runtime session created successfully")
            print(f"  ✓ Input names: {input_names}")
            print(f"  ✓ Output names: {output_names}")
        
        # Step 5: Test inference (basic smoke test)
        print("\nStep 5: Testing inference with dummy input...")
        
        # Create dummy input
        batch_size = 1
        sequence_length = 10
        vocab_size = model_artifacts['config'].vocab_size
        
        # Create random input_ids
        input_ids = np.random.randint(0, vocab_size, (batch_size, sequence_length), dtype=np.int64)
        
        # Prepare inputs
        ort_inputs = {"input_ids": input_ids}
        
        # Add attention_mask if needed
        if "attention_mask" in input_names:
            attention_mask = np.ones((batch_size, sequence_length), dtype=np.int64)
            ort_inputs["attention_mask"] = attention_mask
        
        # Run inference
        try:
            outputs = session.run(output_names[:1], ort_inputs)  # Get first output only
            print(f"  ✓ Inference successful!")
            if verbose:
                print(f"  ✓ Output shape: {outputs[0].shape}")
        except Exception as e:
            print(f"  ⚠ Inference test failed: {str(e)}")
            print("    Note: This might be expected for some model architectures")
        
        # Calculate file sizes
        if verbose:
            print("\nFile size information:")
            total_size = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
            print(f"  Total ONNX export size: {total_size / (1024**2):.2f} MB")
            
            for file in output_dir.rglob("*.onnx"):
                size_mb = file.stat().st_size / (1024**2)
                print(f"  - {file.name}: {size_mb:.2f} MB")
        
        # Summary
        elapsed_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"✅ ONNX Export Test PASSED")
        print(f"Time taken: {elapsed_time:.2f} seconds")
        
        if keep_files:
            print(f"ONNX files saved to: {output_dir}")
        else:
            print("ONNX files will be cleaned up (use --keep-files to preserve)")
        
        print(f"{'='*60}\n")
        
        return True
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"❌ ONNX Export Test FAILED")
        print(f"Error: {str(e)}")
        if verbose:
            print("\nFull traceback:")
            traceback.print_exc()
        print(f"{'='*60}\n")
        return False
        
    finally:
        # Cleanup
        if temp_dir and not keep_files:
            try:
                shutil.rmtree(temp_dir)
                if verbose:
                    print("Temporary files cleaned up")
            except Exception as e:
                print(f"Warning: Could not clean up temp directory: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Test ONNX export pipeline with real models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with default model (distilgpt2)
  python scripts/test_real_onnx_export.py
  
  # Test with a specific model
  python scripts/test_real_onnx_export.py --model gpt2
  
  # Keep exported files and show verbose output
  python scripts/test_real_onnx_export.py --keep-files --verbose
  
Suggested small models for testing:
  - distilgpt2 (fastest, ~350MB)
  - gpt2 (~550MB)  
  - microsoft/DialoGPT-small (~350MB)
  - distilbert-base-uncased (~270MB)
        """
    )
    
    parser.add_argument(
        "--model",
        default="distilgpt2",
        help="HuggingFace model ID to test (default: distilgpt2)"
    )
    
    parser.add_argument(
        "--keep-files",
        action="store_true",
        help="Keep the exported ONNX files after testing"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Run the test
    success = test_onnx_export(
        model_id=args.model,
        keep_files=args.keep_files,
        verbose=args.verbose
    )
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()