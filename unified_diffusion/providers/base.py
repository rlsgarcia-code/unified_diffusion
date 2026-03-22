from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from unified_diffusion.cache.manager import CacheManager, LocalModelRef
from unified_diffusion.registry.models import ModelSpec


@dataclass(slots=True)
class LoadedPipeline:
    pipeline: Any
    device_used: str
    dtype_used: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseProvider(ABC):
    provider_name: str

    @abstractmethod
    def can_handle(self, model_spec: ModelSpec) -> bool:
        raise NotImplementedError

    @abstractmethod
    def ensure_downloaded(
        self,
        model_spec: ModelSpec,
        cache: CacheManager,
        revision: str,
    ) -> LocalModelRef:
        raise NotImplementedError

    @abstractmethod
    def load_pipeline(
        self,
        local_ref: LocalModelRef,
        device: str,
        dtype: str,
        optimizations: dict[str, Any],
    ) -> LoadedPipeline:
        raise NotImplementedError

    @abstractmethod
    def generate(self, pipeline: LoadedPipeline, request: Any) -> list[Image.Image]:
        raise NotImplementedError
