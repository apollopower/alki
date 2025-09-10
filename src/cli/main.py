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
    from ..core.bundle import Bundle
    from ..core.manifest import ManifestGenerator, ModelInfo
    from ..core.model_loader import LlamaCppModelLoader
except ImportError:
    # Handle case when running as a script
    from core.validator import GGUFValidator
    from core.bundle import Bundle
    from core.manifest import ManifestGenerator, ModelInfo
    from core.model_loader import LlamaCppModelLoader

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


@app.command("pack")
def pack(
    model: str = typer.Argument(
        help="GGUF model file path or HuggingFace repo ID (e.g., 'Qwen/Qwen3-0.6B-GGUF')"
    ),
    filename: Optional[str] = typer.Option(
        None,
        "--filename",
        "-f",
        help="Filename pattern for HuggingFace GGUF models (e.g., '*q8_0.gguf')",
    ),
    out: str = typer.Option(
        "./dist",
        "--out",
        "-o",
        help="Output directory for bundle (default: ./dist)",
    ),
    name: Optional[str] = typer.Option(
        None,
        "--name",
        "-n",
        help="Bundle name (derived from model if not provided)",
    ),
    quant: str = typer.Option(
        "Q4_K_M",
        "--quant",
        "-q",
        help="Quantization profile (default: Q4_K_M)",
    ),
    context_size: int = typer.Option(
        2048,
        "--context-size",
        "-c",
        help="Context window size in tokens (default: 2048, extracted from model if available)",
    ),
    no_validate: bool = typer.Option(
        False,
        "--no-validate",
        help="Skip model validation before packing",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
    no_cleanup: bool = typer.Option(
        False,
        "--no-cleanup",
        help="Skip cache cleanup after packing (HuggingFace models only)",
    ),
):
    """
    Create a deployment bundle from a GGUF model.

    This command validates the model (unless skipped), extracts its capabilities,
    and creates a complete deployment bundle with manifests, deployment configs,
    and documentation.

    Examples:
        alki pack /path/to/model.gguf --out ./bundles
        alki pack "Qwen/Qwen3-0.6B-GGUF" --filename "*Q8_0.gguf" --name qwen3-0.6b
        alki pack "Qwen/Qwen2-0.5B-Instruct-GGUF" -f "*q8_0.gguf" --quant Q5_K_M
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    typer.echo("🌊 Alki Pack - Creating deployment bundle...")

    output_dir = Path(out)
    model_path = Path(model) if Path(model).exists() else None

    # Derive bundle name from model if not provided
    if not name:
        if model_path:
            name = model_path.stem
        else:
            # For HuggingFace repos, use the model part of the repo ID
            name = model.split("/")[-1]

    typer.echo(f"Bundle name: {name}")
    typer.echo(f"Output directory: {output_dir}")

    # Step 1: Validate model and get capabilities using our existing validator
    validator = GGUFValidator()
    validation_result = None

    if not no_validate:
        typer.echo("🔍 Validating model and extracting capabilities...")

        try:
            # Use our existing validation infrastructure
            if model_path and model_path.exists():
                # Local file validation
                validation_result = validator.validate_file(
                    str(model_path),
                    max_tokens=10,  # Quick validation test
                    n_ctx=context_size,
                )
            elif filename is not None:
                # HuggingFace repo validation with optional cleanup
                validation_result = validator.validate_and_cleanup(
                    repo_id=model,
                    filename=filename,
                    max_tokens=10,  # Quick validation test
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

            if not validation_result.passed:
                typer.echo(
                    f"❌ Model validation failed: {validation_result.error}", err=True
                )
                raise typer.Exit(1)

            typer.echo("✅ Model validation passed")
            typer.echo(f"  Context length: {validation_result.context_length}")
            typer.echo(f"  Vocabulary size: {validation_result.vocab_size:,}")
            typer.echo(f"  Embedding size: {validation_result.embedding_size}")

        except Exception as e:
            typer.echo(f"❌ Validation error: {e}", err=True)
            raise typer.Exit(1)

    # Step 2: Determine model file path and capabilities
    if model_path and model_path.exists():
        # Local file
        final_model_path = model_path
    else:
        # For HuggingFace models, we need to get the model path from the loader
        # The validation process already downloaded it, but we need to get the path
        try:
            typer.echo(f"📥 Preparing model from HuggingFace: {model}")
            loader = LlamaCppModelLoader()
            # This will reuse the cached download from validation
            model_obj = loader.prepareFromHuggingFace(
                repo_id=model, filename=filename, verbose=verbose, n_ctx=context_size
            )
            # Extract the model path from the llama-cpp model object
            final_model_path = Path(model_obj.model_path)
        except Exception as e:
            typer.echo(f"❌ Failed to prepare model: {e}", err=True)
            raise typer.Exit(1)

    # Step 3: Use validation results for capabilities or extract them
    if validation_result and not no_validate:
        # Use capabilities from validation
        capabilities = {
            "context_length": validation_result.context_length,
            "vocab_size": validation_result.vocab_size,
            "embedding_size": validation_result.embedding_size,
        }
        # Use extracted context length if it's available and larger than our CLI parameter
        model_context = validation_result.context_length
        actual_context = (
            max(context_size, model_context)
            if model_context is not None
            else context_size
        )
    else:
        # Extract capabilities using ManifestGenerator (fallback)
        typer.echo("🔍 Extracting model capabilities...")
        generator = ManifestGenerator()
        capabilities = generator.extract_model_capabilities(final_model_path)

        if capabilities:
            typer.echo(f"  Context length: {capabilities['context_length']}")
            typer.echo(f"  Vocabulary size: {capabilities['vocab_size']}")
            # Use extracted context length if it's available and larger than our CLI parameter
            model_context = capabilities.get("context_length")
            actual_context = (
                max(context_size, model_context)
                if model_context is not None
                else context_size
            )
        else:
            typer.echo("⚠️  Could not extract model capabilities, using CLI defaults")
            capabilities = {"context_length": context_size}
            actual_context = context_size

    # Step 4: Create bundle
    typer.echo("📦 Creating bundle structure...")
    bundle = Bundle(output_dir, name)
    bundle.create_structure()

    # Step 5: Add model to bundle
    typer.echo("➕ Adding model to bundle...")
    try:
        artifact = bundle.add_model(final_model_path, quant)
        typer.echo(f"  Added: {artifact.filename} ({artifact.size:,} bytes)")
    except Exception as e:
        typer.echo(f"❌ Failed to add model to bundle: {e}", err=True)
        raise typer.Exit(1)

    # Step 6: Create model info and manifests
    typer.echo("📝 Generating manifests...")

    model_info = ModelInfo(
        architecture="GGUF",
        context_length=capabilities.get("context_length"),
        vocab_size=capabilities.get("vocab_size"),
        embedding_size=capabilities.get("embedding_size"),
        license="Check original model",
    )

    # Detect chat template
    generator = ManifestGenerator()
    chat_template = generator.detect_chat_template(name)
    typer.echo(f"  Chat template: {chat_template}")

    # Create manifests
    bundle.create_manifest(
        artifacts=[artifact],
        template=chat_template,
        license=model_info.license,
        source_model=final_model_path.name,
        context_size=actual_context,
    )

    bundle.create_runtime_manifest()
    bundle.create_sbom()

    # Step 7: Add documentation
    typer.echo("📄 Creating documentation...")
    bundle.add_readme(name, [quant])
    bundle.add_license(
        "Please add the appropriate license for your model.\n"
        "Check the original model repository for license information."
    )

    # Step 8: Create deployment configs
    typer.echo("🚀 Creating deployment configurations...")

    model_filename = f"{name}-{quant.lower()}.gguf"

    # Systemd service
    systemd_config = generator.create_deployment_placeholder(
        "systemd", name, model_filename, actual_context, chat_template
    )
    (bundle.deploy_dir / "systemd" / f"alki-{name}.service").write_text(systemd_config)

    # Docker
    docker_config = generator.create_deployment_placeholder(
        "docker", name, model_filename, actual_context, chat_template
    )
    (bundle.deploy_dir / "docker" / "Dockerfile").write_text(docker_config)

    # Kubernetes
    k8s_config = generator.create_deployment_placeholder(
        "k3s", name, model_filename, actual_context, chat_template
    )
    (bundle.deploy_dir / "k3s" / "deployment.yaml").write_text(k8s_config)

    # Step 9: Verify bundle
    typer.echo("🔍 Verifying bundle integrity...")
    if bundle.verify_bundle():
        typer.echo("✅ Bundle verification passed")
    else:
        typer.echo("❌ Bundle verification failed", err=True)
        raise typer.Exit(1)

    # Step 10: Show bundle info
    info = bundle.get_info()
    typer.echo("\n🎉 Bundle created successfully!")
    typer.echo(f"  Name: {info['name']}")
    typer.echo(f"  Version: {info['version']}")
    typer.echo(f"  Location: {info['location']}")
    typer.echo(f"  Total size: {info['total_size_mb']:.1f} MB")

    typer.echo("\n🚀 To deploy with llama-server:")
    typer.echo(f"  llama-server -m {bundle.models_dir}/{model_filename} \\")
    typer.echo(f"    --host 0.0.0.0 --port 8080 --ctx-size {actual_context} \\")
    typer.echo(f"    --chat-format {chat_template}")

    typer.echo("\n📁 Bundle contents:")
    for file_path in sorted(bundle.bundle_dir.rglob("*")):
        if file_path.is_file():
            relative_path = file_path.relative_to(bundle.bundle_dir)
            typer.echo(f"  {relative_path}")

    raise typer.Exit(0)
