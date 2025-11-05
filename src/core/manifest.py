"""
Manifest Generation for Alki Bundles

Handles generation of manifest files including model metadata,
runtime configurations, and deployment specifications.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from llama_cpp import Llama

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Model information for manifest"""

    architecture: str
    context_length: Optional[int] = None
    vocab_size: Optional[int] = None
    embedding_size: Optional[int] = None
    license: Optional[str] = None


class ManifestGenerator:
    """Generates various manifest files for bundles"""

    def __init__(self):
        """Initialize manifest generator"""
        # Chat template mappings - critical for proper model inference
        # Each model family uses different special tokens and formatting
        # llama.cpp needs this to properly format conversations
        self.template_mappings = {
            "qwen": "chatml",  # ChatML format used by Qwen models
            "qwen2": "chatml",
            "qwen3": "chatml",
            "llama": "llama3",  # Llama 3 format with header tags
            "llama2": "llama2",  # Llama 2 format with [INST] tags
            "llama3": "llama3",
            "mistral": "mistral",  # Mistral format with [INST] tags
            "phi": "phi3",  # Phi-3 specific format
            "phi3": "phi3",
            "gemma": "gemma",  # Gemma specific format
            "tinyllama": "llama2",  # TinyLlama uses Llama 2 format
            "stablelm": "chatml",  # StableLM uses ChatML format
        }

    def detect_chat_template(self, model_name: str) -> str:
        """
        Detect chat template based on model name.

        The chat template defines how conversations should be formatted
        for the model (e.g., special tokens, role markers). This is critical
        for proper inference - using the wrong template will cause poor outputs.

        Args:
            model_name: Name of the model

        Returns:
            Template identifier for llama.cpp's --chat-format parameter
        """
        model_lower = model_name.lower()

        for key, template in self.template_mappings.items():
            if key in model_lower:
                logger.debug(
                    f"Detected chat template '{template}' for model '{model_name}'"
                )
                return template

        logger.warning(
            f"No chat template detected for '{model_name}', using 'chatml' as default"
        )
        return "chatml"  # ChatML is a reasonable default

    def generate_model_manifest(
        self,
        name: str,
        version: str,
        artifacts: List[Dict[str, Any]],
        model_info: Optional[ModelInfo] = None,
        chat_template: Optional[str] = None,
        runtime_defaults: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate model manifest with only essential fields for v1

        Args:
            name: Model/bundle name
            version: Bundle version
            artifacts: List of artifact dictionaries (with sha256, size, uri)
            model_info: Optional model information (actual capabilities)
            chat_template: Chat template format (auto-detected if not provided)
            runtime_defaults: Runtime parameters (from actual model)

        Returns:
            Model manifest dictionary
        """
        # Auto-detect chat template if not provided
        if chat_template is None:
            chat_template = self.detect_chat_template(name)

        # Build manifest with only essential fields
        manifest = {
            "name": name,
            "version": version,
            "created_at": datetime.now().isoformat(),
            "artifacts": artifacts,
            "chat_template": chat_template,  # Essential for proper inference
        }

        # Add runtime defaults if provided (based on actual model)
        if runtime_defaults:
            manifest["defaults"] = runtime_defaults
        elif model_info and model_info.context_length:
            # Minimal defaults from model info
            manifest["defaults"] = {
                "ctx": model_info.context_length,
                "threads": "auto",
                "ngl": 0,
            }

        # Add essential model info if available
        if model_info:
            manifest["model_info"] = {}
            if model_info.context_length:
                manifest["model_info"]["context_length"] = model_info.context_length
            if model_info.license:
                manifest["model_info"]["license"] = model_info.license

        logger.info(f"Generated model manifest for {name} v{version}")
        return manifest

    def generate_runtime_manifest(
        self,
        runtime: str = "llama.cpp",
        server_config: Optional[Dict[str, Any]] = None,
        model_args: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate minimal runtime configuration manifest for v1

        Args:
            runtime: Runtime engine identifier
            server_config: Server configuration
            model_args: Model loading arguments (from actual model)

        Returns:
            Runtime manifest dictionary
        """
        # Essential server defaults
        if server_config is None:
            server_config = {"host": "0.0.0.0", "port": 8080, "api": True}

        # Model args should be provided based on actual model
        if model_args is None:
            model_args = {"threads": "auto", "ngl": 0}
            logger.warning(
                "No model args provided. Context size must be set at runtime."
            )

        manifest = {
            "runtime": runtime,
            "version": "1.0",
            "server": server_config,
            "args": model_args,
        }

        # Add warning if context size missing
        if "ctx" not in model_args:
            manifest["warnings"] = [
                "Context size (ctx) not specified. Must be set at runtime."
            ]

        logger.info(f"Generated runtime manifest for {runtime}")
        return manifest

    def extract_model_capabilities(self, model_path: Path) -> Optional[Dict[str, Any]]:
        """
        Extract essential capabilities from a GGUF model file

        Args:
            model_path: Path to GGUF model file

        Returns:
            Dictionary of model capabilities or None if extraction fails
        """
        model = None
        try:
            # Load model with minimal resources just to extract metadata
            model = Llama(
                model_path=str(model_path),
                n_ctx=512,  # Minimal context for metadata extraction
                verbose=False,
                n_threads=1,
            )

            capabilities = {
                "context_length": model.n_ctx_train(),  # Training context length
                "vocab_size": model.n_vocab(),
                "embedding_size": model.n_embd(),
            }

            logger.info(
                f"Extracted capabilities from {model_path.name}: ctx={capabilities['context_length']}"
            )
            return capabilities

        except Exception as e:
            logger.error(f"Failed to extract model capabilities: {e}")
            return None
        finally:
            if model is not None:
                del model

    def create_deployment_placeholder(
        self,
        target: str,
        bundle_name: str,
        model_filename: str,
        context_size: Optional[int] = None,
        chat_template: Optional[str] = None,
    ) -> str:
        """
        Create minimal placeholder deployment configuration for v1

        Args:
            target: Deployment target (systemd, k3s, docker)
            bundle_name: Name of the bundle
            model_filename: Model filename within bundle
            context_size: Actual context size from model (if known)
            chat_template: Chat template format (if known)

        Returns:
            Placeholder content as string
        """
        ctx_arg = (
            f"--ctx-size {context_size}"
            if context_size
            else "--ctx-size <REPLACE_WITH_MODEL_CTX>"
        )
        chat_arg = f"--chat-format {chat_template}" if chat_template else ""

        if target == "systemd":
            return f"""# Systemd service for {bundle_name}
[Unit]
Description=Alki {bundle_name} LLM Service
After=network.target

[Service]
Type=simple
ExecStart=/usr/local/bin/llama-server -m /opt/alki/{bundle_name}/models/{model_filename} --host 0.0.0.0 --port 8080 {ctx_arg} {chat_arg}
Restart=always

[Install]
WantedBy=multi-user.target
"""

        elif target == "docker":
            ctx_env = str(context_size) if context_size else "<REPLACE_WITH_MODEL_CTX>"
            chat_env = (
                f"ENV LLAMA_ARG_CHAT_FORMAT={chat_template}\n" if chat_template else ""
            )
            return f"""FROM ghcr.io/ggml-org/llama.cpp:server

COPY models/{model_filename} /models/{model_filename}

ENV LLAMA_ARG_MODEL=/models/{model_filename}
ENV LLAMA_ARG_CTX_SIZE={ctx_env}
{chat_env}ENV LLAMA_ARG_HOST=0.0.0.0
ENV LLAMA_ARG_PORT=8080

EXPOSE 8080
"""

        elif target == "k3s":
            # Generate complete Kubernetes manifests for deployment
            ctx_env = str(context_size) if context_size else "4096"
            chat_format_value = chat_template or "chatml"

            return f"""# Kubernetes manifests for {bundle_name}
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: {bundle_name}-config
  labels:
    app: {bundle_name}
    component: llm-server
data:
  model-filename: "{model_filename}"
  context-size: "{ctx_env}"
  chat-format: "{chat_format_value}"
  
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: {bundle_name}-model-pvc
  labels:
    app: {bundle_name}
    component: llm-server
spec:
  accessModes:
    - ReadOnlyMany
  resources:
    requests:
      storage: 2Gi  # Adjust based on model size
  # Note: Model must be loaded into PV separately via:
  # 1. initContainer with model download
  # 2. Manual kubectl cp
  # 3. CI/CD pipeline model loading
  
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {bundle_name}
  labels:
    app: {bundle_name}
    component: llm-server
spec:
  replicas: 1
  selector:
    matchLabels:
      app: {bundle_name}
  template:
    metadata:
      labels:
        app: {bundle_name}
        component: llm-server
    spec:
      containers:
      - name: llama-server
        image: ghcr.io/ggml-org/llama.cpp:server
        ports:
        - containerPort: 8080
          name: http
        env:
        - name: LLAMA_ARG_MODEL
          value: "/models/{model_filename}"
        - name: LLAMA_ARG_HOST
          value: "0.0.0.0"
        - name: LLAMA_ARG_PORT
          value: "8080"
        - name: LLAMA_ARG_CTX_SIZE
          valueFrom:
            configMapKeyRef:
              name: {bundle_name}-config
              key: context-size
        - name: LLAMA_ARG_CHAT_FORMAT
          valueFrom:
            configMapKeyRef:
              name: {bundle_name}-config
              key: chat-format
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /v1/models
            port: 8080
          initialDelaySeconds: 15
          periodSeconds: 5
        volumeMounts:
        - name: model-storage
          mountPath: /models
          readOnly: true
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: {bundle_name}-model-pvc
        # Alternative approaches:
        # 1. Use initContainer to download model from bundle registry
        # 2. Bake model into container image (increases image size)
        # 3. Use hostPath for single-node deployments
          
---
apiVersion: v1
kind: Service
metadata:
  name: {bundle_name}
  labels:
    app: {bundle_name}
    component: llm-server
spec:
  type: ClusterIP
  ports:
  - port: 8080
    targetPort: 8080
    protocol: TCP
    name: http
  selector:
    app: {bundle_name}
"""

        else:
            return f"# {target} deployment - TODO in future release\n"

    def generate_sbom(self, bundle_name: str, bundle_version: str) -> Dict[str, Any]:
        """
        Generate minimal SBOM for v1 - just tracks core dependencies

        Args:
            bundle_name: Name of the bundle
            bundle_version: Bundle version

        Returns:
            SBOM dictionary in SPDX format
        """
        sbom = {
            "spdxVersion": "SPDX-2.3",
            "dataLicense": "CC0-1.0",
            "SPDXID": "SPDXRef-DOCUMENT",
            "name": f"{bundle_name}-sbom",
            "documentNamespace": f"https://alki.ai/sbom/{bundle_name}/{bundle_version}",
            "creationInfo": {
                "created": datetime.now().isoformat(),
                "creators": ["Tool: alki"],
            },
            "packages": [
                {
                    "SPDXID": "SPDXRef-Package",
                    "name": bundle_name,
                    "version": bundle_version,
                    "downloadLocation": "NOASSERTION",
                    "filesAnalyzed": False,
                },
                {
                    "SPDXID": "SPDXRef-Runtime",
                    "name": "llama.cpp",
                    "version": "latest",
                    "downloadLocation": "https://github.com/ggml-org/llama.cpp",
                    "filesAnalyzed": False,
                    "licenseConcluded": "MIT",
                },
            ],
            "relationships": [
                {
                    "spdxElementId": "SPDXRef-DOCUMENT",
                    "relatedSpdxElement": "SPDXRef-Package",
                    "relationshipType": "DESCRIBES",
                },
                {
                    "spdxElementId": "SPDXRef-Package",
                    "relatedSpdxElement": "SPDXRef-Runtime",
                    "relationshipType": "DEPENDS_ON",
                },
            ],
        }

        logger.info(f"Generated SBOM for {bundle_name} v{bundle_version}")
        return sbom

    def save_manifest(self, manifest: Dict[str, Any], output_path: Path) -> None:
        """
        Save manifest to file

        Args:
            manifest: Manifest dictionary
            output_path: Path to save manifest
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(manifest, f, indent=2)

        logger.debug(f"Saved manifest to {output_path}")
