"""
Shared constants for the Alki codebase.

This module centralizes commonly used string literals, file names,
and other constants to improve maintainability and reduce duplication.
"""


# ONNX Runtime Execution Providers
class ExecutionProviders:
    CPU = "CPUExecutionProvider"


# File names and paths
class FileNames:
    BUNDLE_MANIFEST = "bundle.yaml"
    ONNX_MODEL = "model.onnx"
    ONNX_MODEL_ORIGINAL = "model_original.onnx"
    TOKENIZER_DIR = "tokenizer"
    TOKENIZER_CONFIG = "tokenizer_config.json"
    TOKENIZER_JSON = "tokenizer.json"
    TOKENIZER_VOCAB = "vocab.json"
    TOKENIZER_MERGES = "merges.txt"
    TOKENIZER_SPECIAL_TOKENS = "special_tokens_map.json"
    TOKENIZER_VOCAB_TXT = "vocab.txt"


# Default configuration values
class Defaults:
    ONNX_OPSET_VERSION = 14
    BATCH_SIZE = 1
    SEQUENCE_LENGTH = 128
    MAX_SEQUENCE_LENGTH = 512
    CALIBRATION_SAMPLES = 128
    CLI_CALIBRATION_SAMPLES = 64
    CLI_MAX_LENGTH = 256

    # SmoothQuant defaults
    SMOOTHQUANT_ALPHA = 0.5
    MIN_SCALE = 1e-5
    MAX_SCALE = 1e5

    # Runtime generation defaults
    RUNTIME_MAX_TOKENS = 100
    RUNTIME_TEMPERATURE = 1.0
    RUNTIME_TOP_P = 0.9
    RUNTIME_TOP_K = 50


# Model input/output names
class ModelIO:
    INPUT_IDS = "input_ids"
    ATTENTION_MASK = "attention_mask"
    POSITION_IDS = "position_ids"
    PAST_KEY_VALUES = "past_key_values"
    LOGITS = "logits"

    DEFAULT_INPUTS = [INPUT_IDS, ATTENTION_MASK]
    DEFAULT_OUTPUTS = [LOGITS]


# Target deployment types
class Targets:
    CPU = "cpu"


# Optimization presets
class Presets:
    FAST = "fast"
    BALANCED = "balanced"
    SMALL = "small"


# Quantization methods
class QuantizationMethods:
    SMOOTHQUANT_W8A8 = "SmoothQuant W8A8"


# Quantization formats and types
class QuantizationFormats:
    QDQ = "QDQ"
    QINT8 = "QInt8"
    QUINT8 = "QUInt8"


# Known architectures
class SupportedArchitectures:
    SUPPORTED = [
        "gpt2",
        "bert",
        "distilbert",
        "roberta",
        "xlm-roberta",
        "gpt-neo",
        "gpt-j",
        "opt",
        "bloom",
        "t5",
        "bart",
        "llama",
        "tinyllama",
        "mistral",
        "codegen",
        "falcon",
        "phi",
        "stablelm",
        "gemma",
    ]

    UNSUPPORTED = ["mamba", "mixtral", "phi3", "qwen2", "qwen3"]


# Dynamic axes for ONNX export
def get_default_dynamic_axes(use_cache: bool = False) -> dict:
    """Get default dynamic axes configuration for ONNX export."""
    axes = {
        ModelIO.INPUT_IDS: {0: "batch_size", 1: "sequence_length"},
        ModelIO.ATTENTION_MASK: {0: "batch_size", 1: "sequence_length"},
    }

    if use_cache:
        axes[ModelIO.PAST_KEY_VALUES] = {
            0: "batch_size",
            2: "past_sequence_length",
        }

    return axes


# File size constants
class FileSizes:
    MB = 1024 * 1024
    GB = 1024 * MB
    MEMORY_LIMIT_MB = 500  # Models larger than this may hit memory constraints
