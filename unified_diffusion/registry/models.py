from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from unified_diffusion.errors import ModelNotFoundError, RegistryValidationError


@dataclass(frozen=True, slots=True)
class ModelSpec:
    canonical_id: str
    provider: str
    source: str
    pipeline_type: str
    default_revision: str | None = None
    license_hint: str | None = None
    notes: str | None = None


# Default built-in model sources live here.
# If an upstream Hugging Face repo_id/revision changes or breaks, update the
# corresponding ModelSpec.source/default_revision entry in this registry.
MODEL_REGISTRY: dict[str, ModelSpec] = {
    "sdxl.base": ModelSpec(
        canonical_id="sdxl.base",
        provider="diffusers",
        source="stabilityai/stable-diffusion-xl-base-1.0",
        pipeline_type="stable-diffusion-xl",
        default_revision="main",
        license_hint="Verify Stability AI SDXL license terms before commercial use.",
        notes="Base SDXL text-to-image pipeline.",
    ),
    "sdxl.refiner": ModelSpec(
        canonical_id="sdxl.refiner",
        provider="diffusers",
        source="stabilityai/stable-diffusion-xl-refiner-1.0",
        pipeline_type="stable-diffusion-xl",
        default_revision="main",
        license_hint="Verify Stability AI SDXL license terms before commercial use.",
        notes="SDXL refiner pipeline.",
    ),
    "sd21.base": ModelSpec(
        canonical_id="sd21.base",
        provider="diffusers",
        source="stabilityai/stable-diffusion-2-1-base",
        pipeline_type="stable-diffusion",
        default_revision="main",
        license_hint="Verify Stability AI SD 2.1 license terms before use.",
        notes="Stable Diffusion 2.1 base model for general text-to-image generation.",
    ),
    "playground.v25": ModelSpec(
        canonical_id="playground.v25",
        provider="diffusers",
        source="playgroundai/playground-v2.5-1024px-aesthetic",
        pipeline_type="stable-diffusion-xl",
        default_revision="main",
        license_hint="Check Playground v2.5 license and model card before use.",
        notes="Playground v2.5 aesthetic general-purpose 1024px model.",
    ),
    "pixart.alpha": ModelSpec(
        canonical_id="pixart.alpha",
        provider="diffusers",
        source="PixArt-alpha/PixArt-XL-2-1024-MS",
        pipeline_type="pixart-alpha",
        default_revision="main",
        license_hint="Check the PixArt model card and license before use.",
        notes="PixArt Alpha via diffusers when supported by the installed version.",
    ),
    "pixart.sigma": ModelSpec(
        canonical_id="pixart.sigma",
        provider="diffusers",
        source="PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
        pipeline_type="pixart-sigma",
        default_revision="main",
        license_hint="Check the PixArt Sigma model card and license before use.",
        notes="PixArt Sigma via diffusers when supported by the installed version.",
    ),
    "sd3.medium": ModelSpec(
        canonical_id="sd3.medium",
        provider="diffusers",
        source="stabilityai/stable-diffusion-3-medium-diffusers",
        pipeline_type="stable-diffusion-3",
        default_revision="main",
        license_hint="Check Stability AI SD3 license requirements before use.",
        notes="Requires a diffusers version with Stable Diffusion 3 pipeline support.",
    ),
    "flux.1-dev": ModelSpec(
        canonical_id="flux.1-dev",
        provider="flux",
        source="black-forest-labs/FLUX.1-dev",
        pipeline_type="flux",
        default_revision="main",
        license_hint="Check FLUX.1 license terms before use.",
        notes="Uses diffusers FluxPipeline when the installed environment supports FLUX.",
    ),
}

CUSTOM_REGISTRY_ENV = "UNIFIED_DIFFUSION_REGISTRY_PATH"
VALID_PROVIDERS = {"diffusers", "flux"}
VALID_PIPELINE_TYPES = {
    "stable-diffusion",
    "stable-diffusion-xl",
    "stable-diffusion-3",
    "pixart-alpha",
    "pixart-sigma",
    "flux",
}


def _validated_model_spec(item: dict[str, object], path: Path) -> ModelSpec:
    canonical_id = str(item.get("canonical_id", "")).strip()
    if not canonical_id:
        raise RegistryValidationError(
            f"Registry file '{path}' has an entry with missing canonical_id."
        )

    provider = str(item.get("provider", "")).strip()
    if provider not in VALID_PROVIDERS:
        raise RegistryValidationError(
            f"Registry entry '{canonical_id}' in '{path}' has unsupported provider '{provider}'. "
            f"Use one of: {sorted(VALID_PROVIDERS)}."
        )

    source = str(item.get("source", "")).strip()
    if not source:
        raise RegistryValidationError(
            f"Registry entry '{canonical_id}' in '{path}' is missing source."
        )

    pipeline_type = str(item.get("pipeline_type", "")).strip()
    if pipeline_type not in VALID_PIPELINE_TYPES:
        raise RegistryValidationError(
            f"Registry entry '{canonical_id}' in '{path}' has unsupported "
            f"pipeline_type '{pipeline_type}'. "
            f"Use one of: {sorted(VALID_PIPELINE_TYPES)}."
        )

    return ModelSpec(
        canonical_id=canonical_id,
        provider=provider,
        source=source,
        pipeline_type=pipeline_type,
        default_revision=_optional_string(item.get("default_revision")),
        license_hint=_optional_string(item.get("license_hint")),
        notes=_optional_string(item.get("notes")),
    )


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _load_custom_registry() -> dict[str, ModelSpec]:
    registry_path = os.environ.get(CUSTOM_REGISTRY_ENV)
    if not registry_path:
        return {}

    path = Path(registry_path).expanduser()
    if not path.exists():
        raise ModelNotFoundError(
            f"Custom registry file '{path}' configured via {CUSTOM_REGISTRY_ENV} was not found."
        )

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        items = []
        for key, value in payload.items():
            entry = dict(value)
            entry.setdefault("canonical_id", key)
            items.append(entry)
    else:
        raise ModelNotFoundError(
            f"Custom registry file '{path}' must contain a JSON object "
            "or JSON array of model specs."
        )

    custom_registry: dict[str, ModelSpec] = {}
    for item in items:
        spec = _validated_model_spec(item, path)
        custom_registry[spec.canonical_id] = spec
    return custom_registry


def _combined_registry() -> dict[str, ModelSpec]:
    registry = dict(MODEL_REGISTRY)
    registry.update(_load_custom_registry())
    return registry


def list_model_ids() -> list[str]:
    return sorted(_combined_registry())


def get_model_spec(model_id: str, provider: str | None = None) -> ModelSpec:
    registry = _combined_registry()
    try:
        spec = registry[model_id]
    except KeyError as exc:
        available = ", ".join(list_model_ids())
        raise ModelNotFoundError(
            f"Unknown model '{model_id}'. Use Diffusion.list_models() "
            f"to inspect supported ids: {available}."
        ) from exc
    if provider is not None and provider != spec.provider:
        raise ModelNotFoundError(
            f"Model '{model_id}' is registered for provider '{spec.provider}', not '{provider}'."
        )
    return spec
