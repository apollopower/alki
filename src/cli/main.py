"""
Alki CLI - Edge LLM Deployment Toolchain

Main command-line interface for the Alki toolchain that converts HuggingFace
models into optimized deployment bundles for edge devices.

Commands:
- build: Create deployment bundle from HuggingFace model
- info: Inspect existing bundles
- list: List available bundles in directory
"""

import sys
from pathlib import Path
import logging

import typer
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

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
from src.core.bundle_builder import create_bundle_from_pipeline
from src.core.bundle_manager import BundleManager, load_bundle, discover_bundles
from src.core.constants import Defaults, Targets, Presets

# Initialize Typer app
app = typer.Typer(
    name="alki",
    help="🌊 Alki - Open-source toolchain for deploying LLMs at the edge",
    no_args_is_help=True,
)

# Initialize Rich console for pretty output
console = Console()


# Configure logging
def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    # Suppress some noisy loggers
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("optimum").setLevel(logging.WARNING)


@app.command()
def build(
    model: str = typer.Argument(..., help="HuggingFace model ID (e.g., 'gpt2')"),
    output: Path = typer.Option(
        Path("./dist"), "--output", "-o", help="Output directory for bundle"
    ),
    target: str = typer.Option(
        Targets.CPU,
        "--target",
        "-t",
        help="Target deployment type",
        click_type=click.Choice([Targets.CPU, Targets.OPENVINO]),
    ),
    preset: str = typer.Option(
        Presets.BALANCED,
        "--preset",
        "-p",
        help="Optimization preset",
        click_type=click.Choice([Presets.FAST, Presets.BALANCED, Presets.SMALL]),
    ),
    quantize: bool = typer.Option(
        True, "--quantize/--no-quantize", help="Apply SmoothQuant W8A8 quantization"
    ),
    alpha: float = typer.Option(
        Defaults.SMOOTHQUANT_ALPHA,
        "--alpha",
        "-a",
        help="SmoothQuant alpha parameter (0.0-1.0)",
        min=0.0,
        max=1.0,
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
):
    """
    Build a deployment bundle from a HuggingFace model.

    This command orchestrates the complete pipeline:
    1. Download model from HuggingFace
    2. Export to ONNX format
    3. Apply quantization (if enabled)
    4. Package into deployment bundle
    """
    setup_logging(verbose)

    # Create output directory name based on model and target
    model_name = model.split("/")[-1]  # Get just the model name, not org/
    bundle_name = f"{model_name}-{target}"
    final_output_path = output / bundle_name

    console.print(
        Panel(
            f"🚀 Building bundle for [bold blue]{model}[/bold blue]\n"
            f"Target: [bold green]{target}[/bold green]\n"
            f"Preset: [bold yellow]{preset}[/bold yellow]\n"
            f"Quantization: [bold magenta]{'Enabled' if quantize else 'Disabled'}[/bold magenta]\n"
            f"Output: [bold cyan]{final_output_path}[/bold cyan]",
            title="🌊 Alki Build",
            border_style="blue",
        )
    )

    try:
        # Step 1: Load model
        console.print(
            "\n[bold blue]Step 1:[/bold blue] Loading model from HuggingFace..."
        )
        loader = HuggingFaceModelLoader()
        model_artifacts = loader.prepare(model)
        console.print(f"✓ Loaded {model_artifacts['architecture']}")

        # Step 2: Export to ONNX
        console.print("\n[bold blue]Step 2:[/bold blue] Exporting to ONNX...")

        onnx_config = OnnxExportConfig(
            use_gpu=(target == Targets.GPU),
            use_cache=False,
            optimize=True,
        )

        exporter = OnnxExporter(onnx_config)

        # Create temporary directory for ONNX output
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_onnx_path = Path(temp_dir) / "onnx_output"
            onnx_artifacts = exporter.export(model_artifacts, temp_onnx_path)
            console.print("✓ ONNX export complete")

            # Step 3: Quantization (optional)
            quantization_artifacts = None
            if quantize:
                console.print(
                    f"\n[bold blue]Step 3:[/bold blue] Applying SmoothQuant quantization (α={alpha})..."
                )

                quant_config = SmoothQuantConfig(
                    alpha=alpha,
                    calibration_samples=Defaults.CLI_CALIBRATION_SAMPLES,
                    per_channel=True,
                    symmetric=True,
                )

                calibration_texts = create_default_calibration_texts()
                tokenizer = model_artifacts["tokenizer"]
                calibration_data = CalibrationDataGenerator(
                    tokenizer,
                    calibration_texts,
                    max_length=Defaults.CLI_MAX_LENGTH,
                )

                # Quantize model
                quantizer = SmoothQuantizer(quant_config)
                quantized_model_path = Path(temp_dir) / "quantized_model.onnx"

                try:
                    quantizer.quantize_model(
                        onnx_artifacts["onnx_path"],
                        quantized_model_path,
                        calibration_data,
                    )

                    quantization_artifacts = {
                        "quantized_model_path": quantized_model_path,
                        "config": quant_config,
                    }
                    console.print("✓ Quantization complete")
                except Exception as e:
                    _handle_quantization_error(e, console)
                    raise RuntimeError(f"Quantization failed: {str(e)}")
            else:
                console.print(
                    "\n[bold yellow]Step 3:[/bold yellow] Skipping quantization"
                )

            # Step 4: Create bundle
            console.print(
                "\n[bold blue]Step 4:[/bold blue] Creating deployment bundle..."
            )

            bundle = create_bundle_from_pipeline(
                model_artifacts=model_artifacts,
                onnx_artifacts=onnx_artifacts,
                quantization_artifacts=quantization_artifacts,
                output_path=final_output_path,
                target=target,
                preset=preset,
            )

            # Display bundle information
            _display_bundle_info(bundle)

    except Exception as e:
        console.print(f"[bold red]❌ Build failed:[/bold red] {str(e)}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def info(
    bundle_path: Path = typer.Argument(
        ..., help="Path to bundle directory or bundle.yaml"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed information"
    ),
):
    """
    Display information about a deployment bundle.
    """
    setup_logging(verbose)

    try:
        bundle = load_bundle(bundle_path)
        _display_bundle_info(bundle, detailed=True)

        if verbose:
            # Show compatibility information
            manager = BundleManager()
            compatibility = manager.validate_runtime_compatibility(bundle)

            console.print("\n[bold blue]Runtime Compatibility:[/bold blue]")
            if compatibility["compatible"]:
                console.print("✓ [green]Compatible[/green]")
            else:
                console.print("❌ [red]Issues found[/red]")
                for issue in compatibility["issues"]:
                    console.print(f"  • {issue}")

            if compatibility["recommendations"]:
                console.print("\n[bold yellow]Recommendations:[/bold yellow]")
                for rec in compatibility["recommendations"]:
                    console.print(f"  • {rec}")

    except Exception as e:
        console.print(f"[bold red]❌ Failed to load bundle:[/bold red] {str(e)}")
        raise typer.Exit(1)


@app.command()
def list(
    search_path: Path = typer.Option(
        Path("./dist"), "--path", "-p", help="Directory to search for bundles"
    ),
    recursive: bool = typer.Option(
        True, "--recursive/--no-recursive", help="Search subdirectories"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed information"
    ),
):
    """
    List all deployment bundles in a directory.
    """
    setup_logging(verbose)

    try:
        bundles = discover_bundles(search_path, recursive)

        if not bundles:
            console.print(f"No bundles found in {search_path}")
            return

        # Create table
        table = Table(title=f"🌊 Bundles in {search_path}")
        table.add_column("Model", style="cyan")
        table.add_column("Target", style="green")
        table.add_column("Preset", style="yellow")
        table.add_column("Size", style="blue")
        table.add_column("Created", style="magenta")

        if verbose:
            table.add_column("Quantization", style="red")
            table.add_column("Path", style="dim")

        for bundle in bundles:
            metadata = bundle.metadata

            # Format size information
            if metadata.quantized_size_mb:
                size_str = f"{metadata.quantized_size_mb}MB"
                if metadata.compression_ratio:
                    reduction = (1 - metadata.compression_ratio) * 100
                    size_str += f" (-{reduction:.1f}%)"
            elif metadata.original_size_mb:
                size_str = f"{metadata.original_size_mb}MB"
            else:
                size_str = "Unknown"

            # Format created date
            created_str = metadata.created_at.strftime("%Y-%m-%d %H:%M")

            row = [
                metadata.model_id,
                metadata.target,
                metadata.preset,
                size_str,
                created_str,
            ]

            if verbose:
                quant_str = metadata.quantization_method or "None"
                path_str = str(bundle.bundle_path) if bundle.bundle_path else "Unknown"
                row.extend([quant_str, path_str])

            table.add_row(*row)

        console.print(table)
        console.print(f"\nFound {len(bundles)} bundle(s)")

    except Exception as e:
        console.print(f"[bold red]❌ Failed to list bundles:[/bold red] {str(e)}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


def _display_bundle_info(bundle, detailed: bool = False):
    """Helper function to display bundle information."""
    metadata = bundle.metadata
    runtime = bundle.runtime_config

    # Create info panel
    info_text = f"[bold blue]Model:[/bold blue] {metadata.model_id}\n"
    info_text += f"[bold blue]Architecture:[/bold blue] {metadata.architecture}\n"
    info_text += f"[bold blue]Target:[/bold blue] {metadata.target}\n"
    info_text += f"[bold blue]Provider:[/bold blue] {runtime.provider}\n"

    if metadata.quantization_method:
        info_text += (
            f"[bold blue]Quantization:[/bold blue] {metadata.quantization_method}"
        )
        if metadata.quantization_alpha is not None:
            info_text += f" (α={metadata.quantization_alpha})"
        info_text += "\n"

    if metadata.original_size_mb and metadata.quantized_size_mb:
        reduction = (1 - metadata.compression_ratio) * 100
        info_text += f"[bold blue]Size:[/bold blue] {metadata.original_size_mb}MB → {metadata.quantized_size_mb}MB ({reduction:.1f}% reduction)\n"
    elif metadata.original_size_mb:
        info_text += f"[bold blue]Size:[/bold blue] {metadata.original_size_mb}MB\n"

    info_text += f"[bold blue]Created:[/bold blue] {metadata.created_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
    info_text += f"[bold blue]Alki Version:[/bold blue] {metadata.alki_version}"

    if bundle.bundle_path:
        info_text += f"\n[bold blue]Path:[/bold blue] {bundle.bundle_path}"

    console.print(Panel(info_text, title="📦 Bundle Information", border_style="green"))

    # Validate bundle
    issues = bundle.validate()
    if issues:
        console.print(
            Panel(
                "\n".join(f"• {issue}" for issue in issues),
                title="⚠️  Validation Issues",
                border_style="red",
            )
        )
    else:
        console.print("✅ [green]Bundle validation passed[/green]")


def _handle_quantization_error(error: Exception, console: Console) -> None:
    """Handle quantization errors with helpful messages."""
    error_msg = str(error)
    if "No data is collected" in error_msg:
        console.print(
            "❌ [red]Quantization failed: No calibration data collected[/red]"
        )
        console.print(
            "   This may indicate an issue with the calibration data iterator."
        )
    else:
        console.print(f"❌ [red]Quantization failed: {error_msg}[/red]")


if __name__ == "__main__":
    app()
