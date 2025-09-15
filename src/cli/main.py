"""
Alki CLI - Edge LLM Deployment Toolchain

Main command-line interface for the Alki toolchain that converts HuggingFace
models into optimized deployment bundles for edge devices.
"""

import typer
from typing import Optional
import logging
import json
import shutil
from pathlib import Path

try:
    from ..core.validator import GGUFValidator
    from ..core.bundle import Bundle
    from ..core.manifest import ManifestGenerator, ModelInfo
    from ..core.model_loader import LlamaCppModelLoader
    from ..core.image_builder import ImageBuilder
    from ..core.registry_publisher import RegistryPublisher
    from ..core.model_detector import ModelDetector, ModelType
    from ..core.tool_manager import DependencyError, ConversionError
    from ..converters import get_converter
except ImportError:
    # Handle case when running as a script
    from core.validator import GGUFValidator
    from core.bundle import Bundle
    from core.manifest import ManifestGenerator, ModelInfo
    from core.model_loader import LlamaCppModelLoader
    from core.image_builder import ImageBuilder
    from core.registry_publisher import RegistryPublisher
    from core.model_detector import ModelDetector, ModelType
    from core.tool_manager import DependencyError, ConversionError
    from converters import get_converter

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
    quantize: list[str] = typer.Option(
        ["Q4_K_M"],
        "--quantize",
        "-q",
        help="Quantization profiles for HuggingFace model conversion (e.g., Q4_K_M,Q5_K_M)",
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
    Create a deployment bundle from a model (GGUF or HuggingFace).

    Automatically detects model type and converts HuggingFace PyTorch models to GGUF if needed.
    This command validates the model (unless skipped), extracts its capabilities,
    and creates a complete deployment bundle with manifests, deployment configs,
    and documentation.

    Examples:
        # Local GGUF file
        alki pack /path/to/model.gguf --out ./bundles

        # HuggingFace GGUF repository
        alki pack "Qwen/Qwen3-0.6B-GGUF" --filename "*Q8_0.gguf" --name qwen3-0.6b

        # NEW: HuggingFace PyTorch model (auto-converts to GGUF)
        alki pack "meta-llama/Llama-3.2-1B" --quantize Q4_K_M,Q5_K_M
        alki pack "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
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

    # Step 1: Detect model type and source
    detector = ModelDetector()
    model_type = detector.detect_model_type(model, filename)

    typer.echo(f"🔍 Detected model type: {model_type.value}")

    # Step 2: Handle model based on type
    validator = GGUFValidator()
    validation_result = None
    final_model_path = None

    if model_type == ModelType.LOCAL_GGUF:
        # Handle local GGUF files (existing logic)
        final_model_path = model_path

        if not no_validate:
            typer.echo("🔍 Validating local GGUF model...")
            try:
                validation_result = validator.validate_file(
                    str(final_model_path),
                    max_tokens=10,
                    n_ctx=context_size,
                )
                if not validation_result.passed:
                    typer.echo(
                        f"❌ Model validation failed: {validation_result.error}",
                        err=True,
                    )
                    raise typer.Exit(1)
                typer.echo("✅ Model validation passed")
            except Exception as e:
                typer.echo(f"❌ Validation error: {e}", err=True)
                raise typer.Exit(1)

    elif model_type == ModelType.HF_GGUF:
        # Handle HuggingFace GGUF repositories (existing logic)
        if not filename:
            typer.echo(
                "❌ Error: --filename required for HuggingFace GGUF models", err=True
            )
            raise typer.Exit(1)

        if not no_validate:
            typer.echo("🔍 Validating HuggingFace GGUF model...")
            try:
                validation_result = validator.validate_and_cleanup(
                    repo_id=model,
                    filename=filename,
                    max_tokens=10,
                    cleanup=not no_cleanup,
                    n_ctx=context_size,
                )
                if not validation_result.passed:
                    typer.echo(
                        f"❌ Model validation failed: {validation_result.error}",
                        err=True,
                    )
                    raise typer.Exit(1)
                typer.echo("✅ Model validation passed")
            except Exception as e:
                typer.echo(f"❌ Validation error: {e}", err=True)
                raise typer.Exit(1)

        # Get model path from loader
        try:
            typer.echo(f"📥 Preparing model from HuggingFace: {model}")
            loader = LlamaCppModelLoader()
            model_obj = loader.prepareFromHuggingFace(
                repo_id=model, filename=filename, verbose=verbose, n_ctx=context_size
            )
            final_model_path = Path(model_obj.model_path)
        except Exception as e:
            typer.echo(f"❌ Failed to prepare model: {e}", err=True)
            raise typer.Exit(1)

    elif model_type == ModelType.HF_PYTORCH:
        # NEW: Handle HuggingFace PyTorch models with auto-conversion
        typer.echo("🔄 Converting HuggingFace model to GGUF format...")

        try:
            # Get converter
            converter = get_converter("gguf")

            # Check if we can convert
            if not converter.can_convert(model):
                typer.echo(
                    "❌ Cannot convert model: missing dependencies or unsupported architecture",
                    err=True,
                )
                typer.echo(
                    "Install conversion dependencies with: pip install alki[convert]"
                )
                raise typer.Exit(1)

            # Create temporary directory for conversion
            import tempfile

            temp_dir = Path(tempfile.mkdtemp(prefix="alki_convert_"))

            try:
                # Convert the model
                result = converter.convert(
                    source=model,
                    output_dir=temp_dir,
                    quantizations=quantize,
                    cleanup=not no_cleanup,
                    timeout_minutes=30,
                )

                if not result.success:
                    typer.echo(f"❌ Conversion failed: {result.error}", err=True)
                    raise typer.Exit(1)

                # Use the first converted file for packing
                final_model_path = result.output_files[0]
                typer.echo(f"✅ Conversion complete: {final_model_path.name}")

                # Validate the converted GGUF file
                if not no_validate:
                    typer.echo("🔍 Validating converted GGUF model...")
                    validation_result = validator.validate_file(
                        str(final_model_path),
                        max_tokens=10,
                        n_ctx=context_size,
                    )
                    if not validation_result.passed:
                        typer.echo(
                            f"❌ Converted model validation failed: {validation_result.error}",
                            err=True,
                        )
                        raise typer.Exit(1)
                    typer.echo("✅ Converted model validation passed")

            except Exception as conv_error:
                # Clean up temp directory on error
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                raise conv_error

        except DependencyError as e:
            typer.echo(f"❌ Missing dependencies: {e}", err=True)
            typer.echo("Install with: pip install alki[convert]")
            raise typer.Exit(1)
        except ConversionError as e:
            typer.echo(f"❌ Conversion failed: {e}", err=True)
            raise typer.Exit(1)
        except Exception as e:
            typer.echo(f"❌ Unexpected error during conversion: {e}", err=True)
            raise typer.Exit(1)

    else:
        typer.echo(
            f"❌ Unsupported or unknown model type: {model_type.value}", err=True
        )
        raise typer.Exit(1)

    if not final_model_path or not final_model_path.exists():
        typer.echo("❌ No valid model file found after processing", err=True)
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
        # For converted models, use the first quantization; for existing GGUF, use None
        quantization_info = quantize[0] if model_type == ModelType.HF_PYTORCH else None
        artifact = bundle.add_model(final_model_path, quantization_info)
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
    # For converted models, include all quantizations; for existing GGUF, use empty list
    quantization_list = quantize if model_type == ModelType.HF_PYTORCH else []
    bundle.add_readme(name, quantization_list)
    bundle.add_license(
        "Please add the appropriate license for your model.\n"
        "Check the original model repository for license information."
    )

    # Step 8: Create deployment configs
    typer.echo("🚀 Creating deployment configurations...")

    # Use the actual model filename from the artifact
    model_filename = artifact.uri.split("/")[-1]

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


@app.command("publish")
def publish(
    bundle: str = typer.Argument(help="Path to bundle directory"),
    registry: Optional[str] = typer.Option(
        None, "--registry", "-r", help="Registry URL (e.g., 'myregistry.com/bundles')"
    ),
    tag: str = typer.Option(
        "latest", "--tag", "-t", help="Image tag (default: latest)"
    ),
    name: Optional[str] = typer.Option(
        None, "--name", "-n", help="Bundle name (derived from bundle if not provided)"
    ),
    local: bool = typer.Option(
        False, "--local", "-l", help="Build locally only, don't push to registry"
    ),
    username: Optional[str] = typer.Option(
        None,
        "--username",
        "-u",
        help="Registry username (prefer env REGISTRY_USERNAME)",
    ),
    password: Optional[str] = typer.Option(
        None,
        "--password",
        "-p",
        help="Registry password (prefer env REGISTRY_PASSWORD)",
    ),
    no_validate: bool = typer.Option(
        False, "--no-validate", help="Skip bundle validation before publishing"
    ),
    output_manifest: Optional[str] = typer.Option(
        None,
        "--output-manifest",
        "-o",
        help="Save Kubernetes deployment manifest to file",
    ),
    namespace: str = typer.Option(
        "default", "--namespace", help="Kubernetes namespace for deployment manifest"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
):
    """
    Publish bundle to container registry for fleet deployment.
    
    This command creates a container image with the bundle embedded and optionally 
    pushes it to a registry. Can be used for local-only builds or registry publishing.
    
    Examples:
        # Local build only (for testing)
        alki publish ./dist/qwen3-0.6b --local
        
        # Publish to registry (uses existing docker login)
        alki publish ./dist/qwen3-0.6b --registry myregistry.com/bundles --tag v1.0.0
        
        # Publish with environment authentication
        REGISTRY_USERNAME=myuser REGISTRY_PASSWORD=mypass \\
        alki publish ./dist/qwen3-0.6b --registry harbor.company.com/ai-models
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate parameters
    if not local and not registry:
        typer.echo(
            "❌ Error: Either --registry must be specified or --local flag must be used",
            err=True,
        )
        typer.echo("Examples:", err=True)
        typer.echo(
            "  alki publish ./dist/bundle --local                    # Local build only",
            err=True,
        )
        typer.echo(
            "  alki publish ./dist/bundle --registry myregistry.com  # Push to registry",
            err=True,
        )
        raise typer.Exit(1)

    if username and password:
        typer.echo(
            "⚠️  Warning: Providing passwords via command line is insecure. "
            "Consider using environment variables REGISTRY_USERNAME and REGISTRY_PASSWORD instead.",
            err=True,
        )

    mode_desc = "🏠 Local build" if local else "🚀 Registry publish"
    typer.echo(f"{mode_desc} - Processing bundle...")

    bundle_path = Path(bundle)

    # Validate bundle path
    if not bundle_path.exists():
        typer.echo(f"❌ Bundle path does not exist: {bundle_path}", err=True)
        raise typer.Exit(1)

    if not bundle_path.is_dir():
        typer.echo(f"❌ Bundle path is not a directory: {bundle_path}", err=True)
        raise typer.Exit(1)

    # Check for required bundle structure
    manifest_path = bundle_path / "metadata" / "manifest.json"
    if not manifest_path.exists():
        typer.echo(f"❌ Bundle manifest not found: {manifest_path}", err=True)
        typer.echo("This doesn't appear to be a valid Alki bundle directory.", err=True)
        typer.echo("Run 'alki pack' first to create a bundle.", err=True)
        raise typer.Exit(1)

    # Validate bundle if requested
    if not no_validate:
        typer.echo("🔍 Validating bundle structure...")
        try:
            bundle_name = name or bundle_path.name
            validator = Bundle(bundle_path.parent, bundle_name)
            validator.bundle_dir = bundle_path

            if not validator.verify_bundle():
                typer.echo("❌ Bundle validation failed!", err=True)
                raise typer.Exit(1)

            typer.echo("✅ Bundle validation passed")
        except Exception as e:
            typer.echo(f"❌ Bundle validation error: {e}", err=True)
            raise typer.Exit(1)

    # Prepare registry authentication
    registry_auth = None
    if username and password:
        registry_host = (
            registry.split("/")[0] if registry and "/" in registry else registry
        )
        registry_auth = {
            "username": username,
            "password": password,
            "registry": registry_host,
        }

    try:
        publisher = RegistryPublisher()

        typer.echo(f"📦 Bundle: {bundle_path}")
        if not local and registry:
            typer.echo(f"🏢 Registry: {registry}")
        typer.echo(f"🏷️  Tag: {tag}")

        # Publish bundle
        if local:
            typer.echo("🔨 Building bundle container locally...")
        else:
            typer.echo("🔨 Building and pushing bundle container...")

        result = publisher.publish_bundle(
            bundle_path=bundle_path,
            registry=registry,
            name=name,
            tag=tag,
            registry_auth=registry_auth,
            local_only=local,
        )

        if result.success:
            if local:
                typer.echo("✅ Bundle built locally!")
                typer.echo(f"  📍 Local image: {result.bundle_uri}")
                typer.echo("\n🚀 To push to registry later:")
                if registry:
                    new_tag = f"{registry.rstrip('/')}/{name or bundle_path.name}:{tag}"
                    typer.echo(f"  docker tag {result.bundle_uri} {new_tag}")
                    typer.echo(f"  docker push {new_tag}")
                else:
                    typer.echo(
                        f"  docker tag {result.bundle_uri} <registry>/<name>:{tag}"
                    )
                    typer.echo(f"  docker push <registry>/<name>:{tag}")
            else:
                typer.echo("✅ Bundle published successfully!")
                typer.echo(f"  📍 Bundle URI: {result.bundle_uri}")
                if result.registry_digest:
                    typer.echo(f"  🔒 Registry digest: {result.registry_digest}")

            if result.size_mb:
                typer.echo(f"  📏 Image size: {result.size_mb:.1f} MB")
            if result.push_time_seconds:
                typer.echo(f"  ⏱️  Time: {result.push_time_seconds:.1f}s")

            # Load bundle metadata for deployment manifest
            with open(manifest_path) as f:
                bundle_metadata = json.load(f)

            # Generate deployment manifest
            deployment_name = name or bundle_metadata.get("name", bundle_path.name)
            bundle_uri_for_k8s = result.bundle_uri

            deployment_manifest = publisher.generate_deployment_manifest(
                bundle_uri=bundle_uri_for_k8s,
                bundle_name=deployment_name,
                bundle_metadata=bundle_metadata,
                namespace=namespace,
            )

            # Save manifest if requested
            if output_manifest:
                manifest_file = Path(output_manifest)
                manifest_file.write_text(deployment_manifest)
                typer.echo(f"📄 Deployment manifest saved: {manifest_file}")

            if not local:
                typer.echo("\n🚀 To deploy on Kubernetes:")
                if output_manifest:
                    typer.echo(f"  kubectl apply -f {output_manifest}")
                else:
                    typer.echo("  kubectl apply -f <saved-manifest>.yaml")

                typer.echo("\n📋 To update existing deployment:")
                typer.echo(
                    f"  kubectl set image deployment/{deployment_name} bundle-loader={result.bundle_uri}"
                )

                typer.echo("\n🔍 To verify deployment:")
                typer.echo(f"  kubectl get pods -l app={deployment_name}")
                typer.echo(f"  kubectl logs -l app={deployment_name} -c llama-server")

        else:
            operation = "build" if local else "publish"
            typer.echo(f"❌ Bundle {operation} failed!", err=True)
            if result.error:
                typer.echo(f"Error: {result.error}", err=True)
            raise typer.Exit(1)

    except Exception as e:
        operation = "build" if local else "publish"
        typer.echo(f"❌ {operation.title()} error: {e}", err=True)
        raise typer.Exit(1)


# Create image sub-app
image_app = typer.Typer(name="image", help="Container image operations")
app.add_typer(image_app, name="image")


@image_app.command("build")
def image_build(
    bundle: str = typer.Argument(help="Path to bundle directory"),
    tag: str = typer.Option(
        ..., "--tag", "-t", help="Docker image tag (e.g., 'mymodel:latest')"
    ),
    base: str = typer.Option(
        "debian", "--base", "-b", help="Base image type (alpine, ubuntu, debian)"
    ),
    push: bool = typer.Option(
        False, "--push", help="Push image to registry after build"
    ),
    ctx_size: Optional[int] = typer.Option(
        None, "--ctx-size", "-c", help="Override context size for runtime"
    ),
    host: str = typer.Option("0.0.0.0", "--host", help="Host to bind server to"),
    port: int = typer.Option(8080, "--port", "-p", help="Port to bind server to"),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
):
    """
    Build container image from bundle.

    Creates an optimized container image with llama-server runtime
    for deploying GGUF models at the edge.

    Examples:
        alki image build ./dist/qwen3-0.6b --tag mymodel:latest
        alki image build ./bundles/llama --tag myorg/llama:v1 --base alpine --push
        alki image build ./dist/model --tag model:dev --ctx-size 4096
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    bundle_path = Path(bundle)

    # Validate bundle path
    if not bundle_path.exists():
        typer.echo(f"❌ Bundle path does not exist: {bundle_path}", err=True)
        raise typer.Exit(1)

    if not bundle_path.is_dir():
        typer.echo(f"❌ Bundle path is not a directory: {bundle_path}", err=True)
        raise typer.Exit(1)

    # Check for required bundle structure
    manifest_path = bundle_path / "metadata" / "manifest.json"
    if not manifest_path.exists():
        typer.echo(f"❌ Bundle manifest not found: {manifest_path}", err=True)
        typer.echo("This doesn't appear to be a valid Alki bundle directory.")
        raise typer.Exit(1)

    typer.echo("🐳 Alki Image Build - Creating container image...")
    typer.echo(f"Bundle: {bundle_path}")
    typer.echo(f"Tag: {tag}")
    typer.echo(f"Base image: {base}")

    # Prepare runtime config
    runtime_config = {"host": host, "port": port}
    if ctx_size:
        runtime_config["ctx"] = ctx_size

    try:
        # Initialize image builder
        builder = ImageBuilder()

        typer.echo("🔨 Building container image...")
        result = builder.build_image(
            bundle_path=bundle_path,
            tag=tag,
            base_image=base,
            runtime_config=runtime_config,
            push=push,
        )

        if result.success:
            typer.echo("✅ Image built successfully!")
            typer.echo(f"  Tag: {result.image_tag}")
            if result.size_mb:
                typer.echo(f"  Size: {result.size_mb:.1f} MB")
            if result.build_time_seconds:
                typer.echo(f"  Build time: {result.build_time_seconds:.1f}s")

            if push:
                typer.echo("📤 Image pushed to registry")

            typer.echo("\n🚀 To run the image:")
            typer.echo(f"  docker run -p {port}:{port} {tag}")

            typer.echo("\n🔍 Test the API:")
            typer.echo(f"  curl http://localhost:{port}/v1/models")
        else:
            typer.echo("❌ Image build failed!", err=True)
            if result.error:
                typer.echo(f"Error: {result.error}", err=True)
            if result.build_log and verbose:
                typer.echo("Build log:", err=True)
                typer.echo(result.build_log, err=True)
            raise typer.Exit(1)

    except Exception as e:
        typer.echo(f"❌ Image build error: {e}", err=True)
        raise typer.Exit(1)


@image_app.command("list")
def image_list(
    filter_tag: Optional[str] = typer.Option(
        None, "--filter", "-f", help="Filter images by tag pattern"
    ),
):
    """
    List available container images.

    Examples:
        alki image list
        alki image list --filter "mymodel*"
    """
    try:
        builder = ImageBuilder()
        images = builder.list_images(filter_tag)

        if not images:
            typer.echo("No images found.")
            return

        typer.echo("📋 Container Images:")
        typer.echo()

        for image in images:
            repo = image.get("Repository", "")
            tag = image.get("Tag", "")
            image_id = image.get("ID", "")[:12]
            created = image.get("CreatedSince", "")
            size = image.get("Size", "")

            typer.echo(f"  {repo}:{tag}")
            typer.echo(f"    ID: {image_id}")
            typer.echo(f"    Created: {created}")
            typer.echo(f"    Size: {size}")
            typer.echo()

    except Exception as e:
        typer.echo(f"❌ Error listing images: {e}", err=True)
        raise typer.Exit(1)


@image_app.command("test")
def image_test(
    tag: str = typer.Argument(help="Docker image tag to test"),
    timeout: int = typer.Option(60, "--timeout", "-t", help="Test timeout in seconds"),
):
    """
    Test container image by running it and checking health.

    Examples:
        alki image test mymodel:latest
        alki image test myorg/llama:v1 --timeout 120
    """
    typer.echo(f"🧪 Testing image: {tag}")

    try:
        builder = ImageBuilder()
        result = builder.test_image(tag, timeout)

        if result["success"]:
            typer.echo("✅ Image test passed!")
            typer.echo(f"  Container: {result.get('container_id', 'unknown')[:12]}")
            typer.echo(f"  Port: {result.get('port', 'unknown')}")
            typer.echo(
                f"  Health check: {'✅' if result.get('health_check') else '❌'}"
            )
            typer.echo(
                f"  Models endpoint: {'✅' if result.get('models_endpoint') else '❌'}"
            )

            if result.get("models_data"):
                models = result["models_data"].get("data", [])
                if models:
                    typer.echo("  Available models:")
                    for model in models:
                        typer.echo(f"    - {model.get('id', 'unknown')}")
        else:
            typer.echo("❌ Image test failed!", err=True)
            if result.get("error"):
                typer.echo(f"Error: {result['error']}", err=True)
            raise typer.Exit(1)

    except Exception as e:
        typer.echo(f"❌ Test error: {e}", err=True)
        raise typer.Exit(1)
