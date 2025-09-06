"""
Alki CLI - Edge LLM Deployment Toolchain

Main command-line interface for the Alki toolchain that converts HuggingFace
models into optimized deployment bundles for edge devices.
"""

import typer
from typing import Optional
import logging
from pathlib import Path

try:
    from ..core.validator import GGUFValidator
except ImportError:
    # Handle case when running as a script
    from core.validator import GGUFValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = typer.Typer(
    name="alki",
    help="Alki 🌊 - Edge LLM deployment toolchain",
    add_completion=False,
)


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """
    Alki 🌊 - Edge LLM deployment toolchain

    A toolchain for deploying LLMs at the edge with validation, optimization, and packaging capabilities.
    """
    if ctx.invoked_subcommand is None:
        typer.echo("Use 'alki --help' to see available commands.")
        raise typer.Exit(0)


@app.command("validate")
def validate(
    model: str = typer.Argument(
        help="GGUF model file path or HuggingFace repo ID (e.g., 'Qwen/Qwen3-0.6B-GGUF')"
    ),
    filename: Optional[str] = typer.Option(
        None,
        "--filename",
        "-f",
        help="Filename pattern for HuggingFace GGUF models (e.g., '*q8_0.gguf')",
    ),
    max_tokens: int = typer.Option(
        50, "--max-tokens", help="Maximum tokens for inference test"
    ),
    prompt: str = typer.Option(
        "Here is an introduction to Alki Beach in Seattle.",
        "--prompt",
        "-p",
        help="Test prompt for inference validation",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
    no_cleanup: bool = typer.Option(
        False,
        "--no-cleanup",
        help="Skip cache cleanup after validation (HuggingFace models only)",
    ),
    context_size: int = typer.Option(
        512,
        "--context-size",
        "-c",
        help="Context window size in tokens (default: 512, max depends on model)",
    ),
):
    """
    Validate a GGUF model to ensure it loads and runs correctly.

    Examples:
        alki validate /path/to/model.gguf
        alki validate "Qwen/Qwen3-0.6B-GGUF" --filename "*q8_0.gguf"
        alki validate "Qwen/Qwen2-0.5B-Instruct-GGUF" -f "*q8_0.gguf" --max-tokens 50
        alki validate "Qwen/Qwen3-0.6B-GGUF" -f "*Q8_0.gguf" --no-cleanup
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    validator = GGUFValidator(test_prompt=prompt)

    # Determine if this is a file path or HuggingFace repo
    if Path(model).exists() and filename is None:
        # Local file validation
        result = validator.validate_file(
            model, max_tokens=max_tokens, n_ctx=context_size
        )
    elif filename is not None:
        # HuggingFace repo validation with optional cleanup
        result = validator.validate_and_cleanup(
            repo_id=model,
            filename=filename,
            max_tokens=max_tokens,
            cleanup=not no_cleanup,
            n_ctx=context_size,
        )
    else:
        typer.echo(
            "Error: For HuggingFace repos, you must specify --filename. "
            "For local files, ensure the file exists.",
            err=True,
        )
        raise typer.Exit(1)

    # Print results
    validator.print_result(result)

    # Exit with appropriate code
    if result.passed:
        typer.echo("\n🎉 Validation completed successfully!")
        raise typer.Exit(0)
    else:
        typer.echo("\n❌ Validation failed!")
        raise typer.Exit(1)
