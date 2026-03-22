from __future__ import annotations

from pathlib import Path
from typing import Any

from unified_diffusion.providers.diffusers_provider import DiffusersProvider
from unified_diffusion.registry.models import ModelSpec


class FluxProvider(DiffusersProvider):
    provider_name = "flux"

    def can_handle(self, model_spec: ModelSpec) -> bool:
        return model_spec.provider == self.provider_name

    def _pipeline_candidates(self, cache_path: Any) -> list[str]:
        if isinstance(cache_path, Path):
            model_key = cache_path.parts[-2] if len(cache_path.parts) >= 2 else ""
        else:
            model_key = getattr(cache_path, "canonical_id", "")
        if model_key == "flux.1-dev":
            return ["FluxPipeline", "AutoPipelineForText2Image"]
        return super()._pipeline_candidates(cache_path)

    def _unsupported_pipeline_message(self, cache_path: Any) -> str:
        if isinstance(cache_path, Path):
            model_key = cache_path.parts[-2] if len(cache_path.parts) >= 2 else "unknown"
        else:
            model_key = getattr(cache_path, "canonical_id", "unknown")
        if model_key == "flux.1-dev":
            return (
                "The installed environment does not expose FluxPipeline. "
                "Install optional dependencies with `pip install unified_diffusion[flux]` "
                "and use a diffusers version that includes FLUX support."
            )
        return super()._unsupported_pipeline_message(cache_path)
