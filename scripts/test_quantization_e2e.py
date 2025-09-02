#!/usr/bin/env python3
"""
Integration test for SmoothQuant W8A8 quantization pipeline.

This script performs end-to-end testing with a real model:
1. Downloads a small model from HuggingFace
2. Exports it to ONNX
3. Applies SmoothQuant quantization
4. Measures size reduction and validates output

Run this manually when you want to test the full pipeline:
    python scripts/test_quantization.py

This is NOT part of the automated test suite due to:
- Model download time (100-500MB)
- Processing time (30-60 seconds)
- Network dependency
"""

import sys
from pathlib import Path
import tempfile
import time
import logging

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.model_loader import HuggingFaceModelLoader
from src.core.onnx_exporter import OnnxExporter, OnnxExportConfig
from src.core.quantizer import (
    SmoothQuantizer,
    SmoothQuantConfig,
    CalibrationDataGenerator,
    create_default_calibration_texts,
)

# Configure logging to see progress
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def format_size(size_bytes: int) -> str:
    """Format size in bytes to human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def test_quantization_pipeline():
    """Run full quantization pipeline test."""
    
    print("\n" + "="*80)
    print("SMOOTHQUANT W8A8 QUANTIZATION TEST")
    print("="*80 + "\n")
    
    # Test configuration
    # Using GPT-2 as it's small (124M params) but realistic
    model_id = "gpt2"
    
    print(f"📦 Test Model: {model_id}")
    print("   (Using GPT-2 as it's small enough for quick testing)\n")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        work_dir = Path(tmpdir)
        
        # Step 1: Load model from HuggingFace
        print("Step 1: Loading model from HuggingFace...")
        loader = HuggingFaceModelLoader()
        model_artifacts = loader.prepare(model_id)
        print(f"   ✓ Model loaded: {model_artifacts['architecture']}")
        print(f"   ✓ Tokenizer ready\n")
        
        # Step 2: Export to ONNX
        print("Step 2: Exporting to ONNX format...")
        exporter = OnnxExporter(OnnxExportConfig(optimize=True))
        onnx_dir = work_dir / "onnx"
        
        start_time = time.time()
        export_result = exporter.export(model_artifacts, onnx_dir)
        export_time = time.time() - start_time
        
        onnx_path = export_result["onnx_path"]
        original_size = onnx_path.stat().st_size
        
        print(f"   ✓ ONNX export complete in {export_time:.1f}s")
        print(f"   ✓ Model size: {format_size(original_size)}\n")
        
        # Step 3: Prepare calibration data
        print("Step 3: Preparing calibration data...")
        tokenizer = model_artifacts["tokenizer"]
        
        # Get diverse calibration texts
        calibration_texts = create_default_calibration_texts()
        # Add a few more domain-specific examples
        calibration_texts.extend([
            "Quantization reduces model size by converting weights to lower precision.",
            "Edge devices have limited memory and compute resources.",
            "The transformer architecture has revolutionized natural language processing.",
        ])
        
        calibration_data = CalibrationDataGenerator(
            tokenizer=tokenizer,
            texts=calibration_texts,
            max_length=128  # Shorter sequences for faster testing
        )
        print(f"   ✓ Created {len(calibration_texts)} calibration samples\n")
        
        # Step 4: Test different quantization configurations
        print("Step 4: Testing quantization configurations...\n")
        
        configs = [
            ("Baseline (α=0.0)", SmoothQuantConfig(alpha=0.0, calibration_samples=32)),
            ("Balanced (α=0.5)", SmoothQuantConfig(alpha=0.5, calibration_samples=32)),
            ("Maximum (α=1.0)", SmoothQuantConfig(alpha=1.0, calibration_samples=32)),
        ]
        
        results = []
        
        for config_name, config in configs:
            print(f"Testing {config_name}:")
            print(f"  • Alpha: {config.alpha}")
            print(f"  • Per-channel: {config.per_channel}")
            print(f"  • Symmetric: {config.symmetric}")
            
            quantizer = SmoothQuantizer(config)
            output_path = work_dir / f"quantized_alpha_{config.alpha}.onnx"
            
            try:
                # Reset calibration data for each configuration
                calibration_data.rewind()
                
                start_time = time.time()
                quantized_path = quantizer.quantize_model(
                    onnx_path,
                    output_path,
                    calibration_data
                )
                quant_time = time.time() - start_time
                
                # Measure results
                quantized_size = quantized_path.stat().st_size
                compression_ratio = original_size / quantized_size
                size_reduction = (1 - quantized_size / original_size) * 100
                
                results.append({
                    "name": config_name,
                    "alpha": config.alpha,
                    "time": quant_time,
                    "original_size": original_size,
                    "quantized_size": quantized_size,
                    "compression_ratio": compression_ratio,
                    "size_reduction": size_reduction,
                    "success": True
                })
                
                print(f"  ✓ Quantization complete in {quant_time:.1f}s")
                print(f"  ✓ Size: {format_size(quantized_size)} ({size_reduction:.1f}% reduction)")
                print(f"  ✓ Compression ratio: {compression_ratio:.1f}x\n")
                
            except Exception as e:
                print(f"  ✗ Quantization failed: {e}\n")
                results.append({
                    "name": config_name,
                    "alpha": config.alpha,
                    "success": False,
                    "error": str(e)
                })
        
        # Step 5: Summary
        print("\n" + "="*80)
        print("QUANTIZATION RESULTS SUMMARY")
        print("="*80 + "\n")
        
        print(f"Original Model: {model_id}")
        print(f"Original Size: {format_size(original_size)}\n")
        
        print("Quantization Results:")
        print("-" * 60)
        print(f"{'Configuration':<20} {'Size':<15} {'Reduction':<12} {'Time':<10}")
        print("-" * 60)
        
        for result in results:
            if result["success"]:
                print(f"{result['name']:<20} "
                      f"{format_size(result['quantized_size']):<15} "
                      f"{result['size_reduction']:.1f}%{'':<10} "
                      f"{result['time']:.1f}s")
            else:
                print(f"{result['name']:<20} FAILED: {result['error'][:30]}")
        
        print("-" * 60)
        
        # Analysis
        print("\n📊 ANALYSIS:")
        print("-" * 40)
        
        successful_results = [r for r in results if r["success"]]
        if successful_results:
            avg_reduction = sum(r["size_reduction"] for r in successful_results) / len(successful_results)
            avg_time = sum(r["time"] for r in successful_results) / len(successful_results)
            
            print(f"• Average size reduction: {avg_reduction:.1f}%")
            print(f"• Average quantization time: {avg_time:.1f}s")
            print(f"• Best compression: {max(r['compression_ratio'] for r in successful_results):.1f}x")
            
            # Educational notes about results
            print("\n📚 WHAT THESE RESULTS MEAN:")
            print("-" * 40)
            
            if avg_reduction > 70:
                print("✓ Excellent compression! >70% size reduction is ideal for edge deployment.")
            elif avg_reduction > 50:
                print("✓ Good compression! 50-70% reduction significantly helps edge deployment.")
            else:
                print("⚠ Modest compression. Consider checking if model is already optimized.")
            
            print("\n• SIZE REDUCTION: We expect ~75% reduction (FP32→INT8 is 4x smaller)")
            print("• ALPHA PARAMETER: Higher alpha handles outliers better but may be slower")
            print("• ACCURACY: Not measured here - requires task-specific evaluation")
            print("• INFERENCE SPEED: INT8 typically 2-4x faster on CPUs with VNNI/AMX")
            
        else:
            print("⚠ All quantization attempts failed. Check error messages above.")
        
        print("\n" + "="*80)
        print("TEST COMPLETE")
        print("="*80 + "\n")
        
        return len(successful_results) > 0


if __name__ == "__main__":
    try:
        success = test_quantization_pipeline()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)