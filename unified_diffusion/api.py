from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from unified_diffusion.cache.manager import CacheManager, resolve_revision
from unified_diffusion.errors import ProviderError
from unified_diffusion.providers.base import BaseProvider
from unified_diffusion.providers.diffusers_provider import DiffusersProvider
from unified_diffusion.providers.flux_provider import FluxProvider
from unified_diffusion.registry.models import ModelSpec, get_model_spec, list_model_ids
from unified_diffusion.telemetry import log_event

VALID_DEVICES = {"mps", "cuda", "cpu"}
VALID_DTYPES = {"fp16", "bf16", "fp32"}


@dataclass(slots=True)
class GenerateRequest:
    model: str
    prompt: str
    negative_prompt: str | None = None
    width: int = 1024
    height: int = 1024
    steps: int = 30
    guidance_scale: float = 7.0
    seed: int | None = None
    num_images: int = 1
    device: str | None = None
    dtype: str | None = None
    provider: str | None = None
    revision: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("GenerateRequest.model must be a non-empty canonical model id.")
        if not self.prompt:
            raise ValueError("GenerateRequest.prompt must be a non-empty string.")
        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                "GenerateRequest.width and GenerateRequest.height "
                "must be positive integers."
            )
        if self.steps <= 0:
            raise ValueError("GenerateRequest.steps must be a positive integer.")
        if self.num_images <= 0:
            raise ValueError("GenerateRequest.num_images must be a positive integer.")
        if self.device is not None and self.device not in VALID_DEVICES:
            raise ValueError(
                f"Unsupported device '{self.device}'. "
                f"Use one of: {sorted(VALID_DEVICES)}."
            )
        if self.dtype is not None and self.dtype not in VALID_DTYPES:
            raise ValueError(
                f"Unsupported dtype '{self.dtype}'. "
                f"Use one of: {sorted(VALID_DTYPES)}."
            )


@dataclass(slots=True)
class GenerateResult:
    images: list[Image.Image]
    seed_used: int | None
    model_resolved: str
    provider_used: str
    cache_path: str
    metadata: dict[str, Any]


class Diffusion:
    def __init__(
        self,
        cache_dir: str | Path = "~/.cache/unified-diffusion",
        log_path: str | Path | None = None,
        default_device: str = "mps",
        default_dtype: str = "fp16",
    ) -> None:
        if default_device not in VALID_DEVICES:
            raise ValueError(
                f"Unsupported default_device '{default_device}'. "
                f"Use one of: {sorted(VALID_DEVICES)}."
            )
        if default_dtype not in VALID_DTYPES:
            raise ValueError(
                f"Unsupported default_dtype '{default_dtype}'. "
                f"Use one of: {sorted(VALID_DTYPES)}."
            )
        self.cache = CacheManager(cache_dir=cache_dir, log_path=log_path)
        self.default_device = default_device
        self.default_dtype = default_dtype

    def list_models(self) -> list[str]:
        return list_model_ids()

    def run(self, request: GenerateRequest) -> GenerateResult:
        log_event("generate.start", model=request.model, provider=request.provider or "auto")
        model_spec = self._resolve_model(request)
        revision = resolve_revision(request.revision, model_spec.default_revision)
        provider = self._get_provider(model_spec.provider)
        local_ref = provider.ensure_downloaded(model_spec, self.cache, revision)

        optimizations = {
            "attention_slicing": request.extra.get("attention_slicing", True),
            "vae_tiling": request.extra.get("vae_tiling", True),
        }
        loaded_pipeline = provider.load_pipeline(
            local_ref=local_ref,
            device=request.device or self.default_device,
            dtype=request.dtype or self.default_dtype,
            optimizations=optimizations,
        )
        images = provider.generate(loaded_pipeline, request)
        resolved_model = f"{model_spec.canonical_id}@{revision}"
        metadata = {
            "source": model_spec.source,
            "license_hint": model_spec.license_hint,
            "notes": model_spec.notes,
            "device_used": loaded_pipeline.device_used,
            "dtype_used": loaded_pipeline.dtype_used,
            **loaded_pipeline.metadata,
        }
        result = GenerateResult(
            images=images,
            seed_used=request.seed,
            model_resolved=resolved_model,
            provider_used=model_spec.provider,
            cache_path=str(local_ref.cache_path),
            metadata=metadata,
        )
        log_event(
            "generate.completed",
            model=result.model_resolved,
            provider=result.provider_used,
            cache_path=result.cache_path,
            image_count=len(result.images),
        )
        return result

    def _resolve_model(self, request: GenerateRequest) -> ModelSpec:
        return get_model_spec(request.model, provider=request.provider)

    def _get_provider(self, provider_name: str) -> BaseProvider:
        providers: dict[str, type[BaseProvider]] = {
            "diffusers": DiffusersProvider,
            "flux": FluxProvider,
        }
        try:
            provider_cls = providers[provider_name]
        except KeyError as exc:
            raise ProviderError(f"Unsupported provider '{provider_name}'.") from exc
        return provider_cls()
