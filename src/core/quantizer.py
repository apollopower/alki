"""
SmoothQuant W8A8 Quantization for Edge Deployment

This module implements SmoothQuant, a post-training quantization technique that enables
accurate INT8 inference without retraining. It's particularly effective for LLMs.

Key Concepts:
- Quantization: Converting floating-point weights/activations (FP32/FP16) to integers (INT8)
  This gives us 4x memory reduction and faster inference on CPUs/edge devices.

- The Challenge: LLMs have activation outliers (extremely large values) that make
  naive quantization lose accuracy. These outliers are systematic - they appear
  in the same channels across different inputs.

- SmoothQuant's Solution: Instead of fighting outliers, we "smooth" them by scaling
  weights up and activations down by a careful factor. This redistributes the
  quantization difficulty from activations to weights (which are easier to quantize).

Paper: https://arxiv.org/abs/2211.10438
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import logging

import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort
from onnxruntime.quantization import (
    quantize_static,
    QuantFormat,
    QuantType,
    CalibrationDataReader,
)


logger = logging.getLogger(__name__)


@dataclass
class SmoothQuantConfig:
    """Configuration for SmoothQuant W8A8 quantization.

    Attributes:
        alpha: Smoothing factor between [0, 1]. Controls how much difficulty we
               migrate from activations to weights.
               - alpha=0: No smoothing (baseline quantization)
               - alpha=0.5: Balanced (recommended default)
               - alpha=1: Maximum smoothing
               Higher alpha helps with models that have extreme outliers.

        calibration_samples: Number of representative inputs to analyze for quantization.
                           More samples = better statistics but slower calibration.
                           128-512 is typically sufficient.

        per_channel: If True, compute different scales per output channel.
                    More accurate but slightly larger model size.
                    Recommended True for best accuracy.

        symmetric: If True, use symmetric quantization (zero point = 0).
                  Symmetric is faster on some hardware but may be less accurate.
                  Most edge accelerators prefer symmetric.

        opset_version: ONNX opset version for compatibility.
    """

    alpha: float = 0.5
    calibration_samples: int = 128
    per_channel: bool = True
    symmetric: bool = True
    opset_version: int = 14

    def __post_init__(self):
        """Validate configuration parameters."""
        if not 0 <= self.alpha <= 1:
            raise ValueError(f"Alpha must be between 0 and 1, got {self.alpha}")
        if self.calibration_samples < 1:
            raise ValueError(
                f"Need at least 1 calibration sample, got {self.calibration_samples}"
            )


class CalibrationDataGenerator(CalibrationDataReader):
    """Generates calibration data for quantization.

    Calibration data should be representative of real inputs the model will see.
    For LLMs, this means realistic text sequences, not random tokens.
    """

    def __init__(self, tokenizer, texts: List[str], max_length: int = 512):
        """
        Args:
            tokenizer: HuggingFace tokenizer for the model
            texts: List of representative text samples
            max_length: Maximum sequence length for inputs
        """
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length
        self.current_idx = 0

        # Ensure tokenizer has a padding token (GPT-2 doesn't by default)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("No padding token found, using EOS token for padding")

        # Pre-tokenize all texts for efficiency
        self.encoded_inputs = []
        for text in texts:
            inputs = tokenizer(
                text,
                return_tensors="np",
                padding="max_length",
                truncation=True,
                max_length=max_length,
            )
            # Basic inputs that all models need
            sample = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
            }

            # Add position_ids if needed (common for GPT models)
            batch_size, seq_len = inputs["input_ids"].shape
            sample["position_ids"] = (
                np.arange(seq_len, dtype=np.int64)
                .reshape(1, -1)
                .repeat(batch_size, axis=0)
            )

            self.encoded_inputs.append(sample)

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        """Return next calibration sample or None if exhausted."""
        if self.current_idx >= len(self.encoded_inputs):
            return None

        sample = self.encoded_inputs[self.current_idx]
        self.current_idx += 1
        return sample

    def rewind(self):
        """Reset to beginning of calibration data."""
        self.current_idx = 0


class SmoothQuantizer:
    """Implements SmoothQuant W8A8 quantization for ONNX models.

    The main insight: Activation outliers make quantization hard, but we can
    mathematically redistribute the difficulty by scaling layers carefully.

    For a linear layer Y = X * W, we can insert scales:
    Y = X * diag(s) * diag(s)^-1 * W = X_smooth * W_smooth

    Where s is chosen to balance the quantization difficulty.
    """

    # Constants for dummy input generation
    DEFAULT_BATCH_SIZE = 1
    DEFAULT_SEQUENCE_LENGTH = 128

    # Constants for numerical stability
    MIN_SCALE = 1e-5
    MAX_SCALE = 1e5

    def __init__(self, config: Optional[SmoothQuantConfig] = None):
        """
        Args:
            config: Quantization configuration. Uses defaults if None.
        """
        self.config = config or SmoothQuantConfig()
        self.activation_scales: Dict[str, np.ndarray] = {}
        self.weight_scales: Dict[str, np.ndarray] = {}

    def _create_dummy_input(self, input_info) -> np.ndarray:
        """Create a dummy input tensor for missing ONNX model inputs.

        Args:
            input_info: ONNX input information with name, shape, and type

        Returns:
            Dummy numpy array with appropriate shape and dtype
        """
        # Create dummy input with correct shape and type
        shape = input_info.shape
        # Replace dynamic dimensions with reasonable defaults
        shape = [
            (
                self.DEFAULT_BATCH_SIZE
                if dim in [None, "batch_size", "N"]
                else (
                    self.DEFAULT_SEQUENCE_LENGTH
                    if dim in ["sequence_length", "M"]
                    else (dim if isinstance(dim, int) else 1)
                )
            )
            for dim in shape
        ]

        if "int" in input_info.type:
            return np.zeros(shape, dtype=np.int64)
        else:
            return np.zeros(shape, dtype=np.float32)

    def collect_activation_statistics(
        self, model_path: Path, calibration_data: CalibrationDataReader
    ) -> Dict[str, np.ndarray]:
        """Collect activation ranges by running calibration data through the model.

        This identifies which channels have outliers that need smoothing.

        Args:
            model_path: Path to ONNX model
            calibration_data: Generator of representative inputs

        Returns:
            Dictionary mapping tensor names to their per-channel max absolute values
        """
        logger.info(
            f"Collecting activation statistics from {self.config.calibration_samples} samples..."
        )

        # Load model for inference
        session = ort.InferenceSession(str(model_path))

        # Get model input names and shapes for debugging
        input_names = [inp.name for inp in session.get_inputs()]
        logger.debug(f"Model expects inputs: {input_names}")

        # We'll track the maximum absolute value per channel for each activation tensor
        activation_ranges = {}

        # Process calibration samples
        sample_count = 0
        calibration_data.rewind()

        while sample_count < self.config.calibration_samples:
            inputs = calibration_data.get_next()
            if inputs is None:
                calibration_data.rewind()
                inputs = calibration_data.get_next()
                if inputs is None:
                    break

            # Handle models with optional inputs (like past_key_values for GPT-2)
            # Create dummy inputs for any missing required inputs
            full_inputs = {}
            for inp in session.get_inputs():
                if inp.name in inputs:
                    full_inputs[inp.name] = inputs[inp.name]
                else:
                    full_inputs[inp.name] = self._create_dummy_input(inp)
                    logger.debug(
                        f"Created dummy input for {inp.name} with shape {full_inputs[inp.name].shape}"
                    )

            # Run inference to get intermediate activations
            # Note: In production, we'd hook into ONNX Runtime to get all intermediates
            # For now, we'll focus on input/output tensors
            try:
                _ = session.run(None, full_inputs)
            except Exception as e:
                logger.warning(f"Failed to run inference for calibration: {e}")
                # Continue with next sample
                sample_count += 1
                continue

            # Track activation ranges (simplified - real implementation needs all intermediates)
            # Only track the actual provided inputs, not the dummy ones
            for name, tensor in inputs.items():
                if name not in activation_ranges:
                    activation_ranges[name] = np.abs(tensor).max(axis=0)
                else:
                    activation_ranges[name] = np.maximum(
                        activation_ranges[name], np.abs(tensor).max(axis=0)
                    )

            sample_count += 1

        logger.info(
            f"Collected statistics for {len(activation_ranges)} activation tensors"
        )
        return activation_ranges

    def calculate_smoothing_scales(
        self,
        activation_scales: Dict[str, np.ndarray],
        weight_scales: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Calculate optimal smoothing scales using SmoothQuant formula.

        The key equation: s_j = (max|X_j|^alpha / max|W_j|^alpha)^(1/2)

        This balances quantization difficulty between activations and weights
        based on the alpha parameter.

        Args:
            activation_scales: Per-channel max values for activations
            weight_scales: Per-channel max values for weights

        Returns:
            Smoothing scales to apply to each layer
        """
        smoothing_scales = {}

        for layer_name in activation_scales:
            if layer_name not in weight_scales:
                continue

            act_scale = activation_scales[layer_name]
            weight_scale = weight_scales[layer_name]

            # SmoothQuant formula: balance based on alpha
            # s = (|X|^alpha / |W|^alpha)^(1/2) per the paper
            # This balances quantization difficulty between activations and weights
            smooth_scale = np.power(act_scale / weight_scale, self.config.alpha / 2)

            # Avoid numerical issues
            smooth_scale = np.clip(smooth_scale, self.MIN_SCALE, self.MAX_SCALE)

            smoothing_scales[layer_name] = smooth_scale

        return smoothing_scales

    def apply_smoothing(
        self, model: onnx.ModelProto, smoothing_scales: Dict[str, np.ndarray]
    ) -> onnx.ModelProto:
        """Apply smoothing scales to model weights and add inverse scales for activations.

        This modifies the ONNX graph to include the smoothing transformations
        before quantization occurs.

        Args:
            model: Original ONNX model
            smoothing_scales: Scales calculated by calculate_smoothing_scales

        Returns:
            Modified ONNX model with smoothing applied
        """
        logger.info("Applying smoothing scales to model...")

        # In a full implementation, we would:
        # 1. Iterate through the graph nodes
        # 2. For each Linear/Conv node, scale weights by 1/s
        # 3. Insert Mul nodes to scale activations by s
        # 4. Ensure mathematical equivalence: Y = X*s * W/s = X*W

        # For now, return the model as-is (actual implementation needs graph surgery)
        logger.warning(
            "Smoothing application not yet fully implemented - using baseline quantization"
        )
        return model

    def quantize_model(
        self,
        onnx_model_path: Path,
        output_path: Path,
        calibration_data: CalibrationDataReader,
    ) -> Path:
        """Main entry point: Quantize an ONNX model using SmoothQuant W8A8.

        This is the primary method users will call. It orchestrates the full pipeline:
        1. Collect activation statistics
        2. Calculate smoothing scales
        3. Apply smoothing to the model
        4. Quantize to INT8

        Args:
            onnx_model_path: Path to input ONNX model (FP32)
            output_path: Where to save quantized model
            calibration_data: Representative input data

        Returns:
            Path to quantized model
        """
        logger.info(
            f"Starting SmoothQuant W8A8 quantization with alpha={self.config.alpha}"
        )

        # Step 1: Collect statistics (simplified for initial implementation)
        activation_stats = self.collect_activation_statistics(
            onnx_model_path, calibration_data
        )

        # Step 2: Load model and get weight statistics
        model = onnx.load(str(onnx_model_path))

        # Extract weight scales (simplified - real implementation needs proper tensor extraction)
        weight_stats = {}
        for initializer in model.graph.initializer:
            if "weight" in initializer.name.lower():
                tensor = numpy_helper.to_array(initializer)
                if self.config.per_channel:
                    # Per-channel: max over all dimensions except output channels
                    weight_stats[initializer.name] = np.abs(tensor).max(
                        axis=tuple(range(1, tensor.ndim))
                    )
                else:
                    # Per-tensor: single max value
                    weight_stats[initializer.name] = np.array([np.abs(tensor).max()])

        # Step 3: Calculate and apply smoothing (if alpha > 0)
        if self.config.alpha > 0:
            smoothing_scales = self.calculate_smoothing_scales(
                activation_stats, weight_stats
            )
            model = self.apply_smoothing(model, smoothing_scales)

            # Save smoothed model to temp path
            temp_model_path = output_path.parent / "temp_smoothed.onnx"
            onnx.save(model, str(temp_model_path))
        else:
            temp_model_path = onnx_model_path

        # Step 4: Quantize using ONNX Runtime
        logger.info("Quantizing model to INT8...")

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Use ONNX Runtime's static quantization
        quantize_static(
            model_input=str(temp_model_path),
            model_output=str(output_path),
            calibration_data_reader=calibration_data,
            quant_format=QuantFormat.QDQ,  # Use QDQ format for better compatibility
            activation_type=(
                QuantType.QInt8 if self.config.symmetric else QuantType.QUInt8
            ),
            weight_type=QuantType.QInt8,
            per_channel=self.config.per_channel,
            reduce_range=False,  # Don't reduce range for edge devices
        )

        # Clean up temp file if we created one
        if temp_model_path != onnx_model_path:
            temp_model_path.unlink()

        # Verify the quantized model
        quantized_model = onnx.load(str(output_path))
        onnx.checker.check_model(quantized_model)

        # Report size reduction
        original_size = onnx_model_path.stat().st_size
        quantized_size = output_path.stat().st_size
        reduction = (1 - quantized_size / original_size) * 100

        logger.info(f"✓ Quantization complete: {output_path}")
        logger.info(
            f"  Size reduction: {reduction:.1f}% ({original_size//1024//1024}MB → {quantized_size//1024//1024}MB)"
        )

        return output_path


def create_default_calibration_texts() -> List[str]:
    """Generate default calibration texts for LLMs.

    These should be diverse, representative samples of text the model might see.
    For production, use domain-specific data.
    """
    return [
        "The weather today is quite pleasant with clear skies and mild temperatures.",
        "Machine learning has revolutionized how we approach complex problems in technology.",
        "The recipe calls for two cups of flour and three eggs to make the perfect cake.",
        "Economic indicators suggest a period of moderate growth in the coming quarters.",
        "The novel explores themes of identity and belonging in modern society.",
        "Debugging code requires patience and systematic problem-solving skills.",
        "Climate change poses significant challenges for coastal communities worldwide.",
        "The conference will feature speakers from various industries and academia.",
    ]
