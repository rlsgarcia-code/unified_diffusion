from __future__ import annotations

import inspect
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from unified_diffusion.cache.manager import CacheManager, LocalModelRef
from unified_diffusion.defaults import DEFAULT_GENERATE_DEVICE
from unified_diffusion.errors import ProviderError
from unified_diffusion.providers.base import BaseProvider, LoadedPipeline
from unified_diffusion.registry.models import ModelSpec


@dataclass(frozen=True, slots=True)
class Optimizations:
    attention_slicing: bool = True
    vae_tiling: bool = True


class DiffusersProvider(BaseProvider):
    provider_name = "diffusers"

    def can_handle(self, model_spec: ModelSpec) -> bool:
        return model_spec.provider == self.provider_name

    def ensure_downloaded(
        self,
        model_spec: ModelSpec,
        cache: CacheManager,
        revision: str,
    ) -> LocalModelRef:
        local_source = Path(model_spec.source).expanduser()
        source_kind = (
            "local_dir"
            if local_source.is_dir()
            else "local_file" if local_source.is_file() else "remote_repo"
        )
        ref = cache.local_ref(
            model_spec.canonical_id,
            revision,
            model_spec.source,
            model_spec.provider,
            model_spec.pipeline_type,
            source_kind=source_kind,
        )
        if cache.is_downloaded(ref):
            return ref

        if source_kind != "remote_repo":
            return self._stage_local_source(local_source, model_spec, cache, ref, revision)

        staging_dir = cache.create_staging_dir(model_spec.canonical_id, revision)
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise ProviderError(
                "huggingface_hub is required to download models. "
                "Install project dependencies first."
            ) from exc

        try:
            with cache.huggingface_env():
                snapshot_download(
                    repo_id=model_spec.source,
                    revision=revision,
                    local_dir=staging_dir,
                    local_dir_use_symlinks=False,
                )
            stats = cache.collect_file_stats(staging_dir)
            cache.finalize_download(
                staging_dir,
                ref,
                metadata={
                    "source": model_spec.source,
                    "provider": model_spec.provider,
                    "revision": revision,
                },
            )
            cache.append_registry_entry(
                {
                    "canonical_id": model_spec.canonical_id,
                    "provider": model_spec.provider,
                    "source": model_spec.source,
                    "revision": revision,
                    "cache_path": str(ref.cache_path),
                    "license_hint": model_spec.license_hint,
                    **stats,
                }
            )
            return ref
        except Exception as exc:
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise ProviderError(
                f"Failed to download model '{model_spec.canonical_id}' "
                f"from '{model_spec.source}': {exc}"
            ) from exc

    def load_pipeline(
        self,
        local_ref: LocalModelRef,
        device: str,
        dtype: str,
        optimizations: dict[str, Any],
    ) -> LoadedPipeline:
        try:
            import torch
        except ImportError as exc:
            raise ProviderError("PyTorch is required to load diffusers pipelines.") from exc

        pipeline_cls = self._resolve_pipeline_class(local_ref)
        target_device, warnings = self._resolve_device(torch, device)
        torch_dtype, resolved_dtype, dtype_warnings = self._resolve_dtype(
            torch,
            dtype,
            target_device,
        )
        warnings.extend(dtype_warnings)

        pipeline = self._load_pipeline_instance(pipeline_cls, local_ref, torch_dtype)

        if hasattr(pipeline, "to"):
            pipeline = pipeline.to(target_device)

        if target_device == "mps" and resolved_dtype == "fp16":
            vae = getattr(pipeline, "vae", None)
            if vae is not None and hasattr(vae, "to"):
                vae.to(dtype=torch.float32)
                warnings.append("Forced VAE to fp32 on MPS to reduce NaN/black-image issues.")

        opts = Optimizations(
            attention_slicing=bool(optimizations.get("attention_slicing", True)),
            vae_tiling=bool(optimizations.get("vae_tiling", True)),
        )
        if opts.attention_slicing and hasattr(pipeline, "enable_attention_slicing"):
            pipeline.enable_attention_slicing()
        vae = getattr(pipeline, "vae", None)
        if opts.vae_tiling and vae is not None and hasattr(vae, "enable_tiling"):
            vae.enable_tiling()

        return LoadedPipeline(
            pipeline=pipeline,
            device_used=target_device,
            dtype_used=resolved_dtype,
            metadata={"warnings": warnings},
        )

    def generate(self, pipeline: LoadedPipeline, request: Any) -> list[Any]:
        call_kwargs = {
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "width": request.width,
            "height": request.height,
            "num_inference_steps": request.steps,
            "guidance_scale": request.guidance_scale,
            "num_images_per_prompt": request.num_images,
        }

        generator = self._make_generator(request.seed, pipeline.device_used, request.num_images)
        if generator is not None:
            call_kwargs["generator"] = generator

        signature = inspect.signature(pipeline.pipeline.__call__)
        accepted = set(signature.parameters)
        filtered_kwargs = {
            key: value
            for key, value in call_kwargs.items()
            if value is not None and key in accepted
        }
        for key, value in request.extra.items():
            if key in accepted:
                filtered_kwargs[key] = value
        try:
            output = pipeline.pipeline(**filtered_kwargs)
        except Exception as exc:
            raise ProviderError(f"Generation failed for model '{request.model}': {exc}") from exc
        images = getattr(output, "images", None)
        if images is None:
            raise ProviderError("The diffusers pipeline did not return an 'images' attribute.")
        return images

    def _resolve_pipeline_class(self, local_ref: LocalModelRef) -> type[Any]:
        try:
            import diffusers
        except ImportError as exc:
            raise ProviderError("diffusers is required for diffusers-backed models.") from exc

        cache_path = local_ref.cache_path
        model_index = cache_path / "model_index.json"
        class_name: str | None = None
        if model_index.exists():
            import json

            with model_index.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            class_name = data.get("_class_name")

        candidates = [class_name, *self._pipeline_candidates(local_ref)]
        for candidate in candidates:
            if not candidate:
                continue
            pipeline_cls = getattr(diffusers, candidate, None)
            if pipeline_cls is not None:
                return pipeline_cls

        raise ProviderError(self._unsupported_pipeline_message(local_ref))

    def _pipeline_candidates(self, local_ref: LocalModelRef | Path) -> list[str]:
        if isinstance(local_ref, Path):
            model_key = local_ref.parts[-2] if len(local_ref.parts) >= 2 else ""
            pipeline_type = None
        else:
            model_key = local_ref.canonical_id
            pipeline_type = local_ref.pipeline_type
        candidates_by_model = {
            "sdxl.base": ["StableDiffusionXLPipeline", "AutoPipelineForText2Image"],
            "sdxl.refiner": ["StableDiffusionXLImg2ImgPipeline", "StableDiffusionXLPipeline"],
            "sd21.base": ["StableDiffusionPipeline", "AutoPipelineForText2Image"],
            "playground.v25": ["StableDiffusionXLPipeline", "AutoPipelineForText2Image"],
            "pixart.alpha": ["PixArtAlphaPipeline", "AutoPipelineForText2Image"],
            "pixart.sigma": [
                "PixArtSigmaPipeline",
                "PixArtAlphaPipeline",
                "AutoPipelineForText2Image",
            ],
            "sd3.medium": ["StableDiffusion3Pipeline", "AutoPipelineForText2Image"],
        }
        candidates_by_pipeline_type = {
            "stable-diffusion": ["StableDiffusionPipeline", "AutoPipelineForText2Image"],
            "stable-diffusion-xl": ["StableDiffusionXLPipeline", "AutoPipelineForText2Image"],
            "stable-diffusion-3": ["StableDiffusion3Pipeline", "AutoPipelineForText2Image"],
            "pixart-alpha": ["PixArtAlphaPipeline", "AutoPipelineForText2Image"],
            "pixart-sigma": [
                "PixArtSigmaPipeline",
                "PixArtAlphaPipeline",
                "AutoPipelineForText2Image",
            ],
            "flux": ["FluxPipeline", "AutoPipelineForText2Image"],
        }
        return candidates_by_model.get(
            model_key,
            candidates_by_pipeline_type.get(
                pipeline_type,
                ["AutoPipelineForText2Image"],
            ),
        )

    def _unsupported_pipeline_message(self, local_ref: LocalModelRef | Path) -> str:
        if isinstance(local_ref, Path):
            model_key = local_ref.parts[-2] if len(local_ref.parts) >= 2 else "unknown"
        else:
            model_key = local_ref.canonical_id
        if model_key == "sd3.medium":
            return (
                "The installed diffusers build does not expose Stable Diffusion 3 support. "
                "Upgrade diffusers to a version that provides StableDiffusion3Pipeline."
            )
        if model_key.startswith("pixart."):
            return (
                "The installed diffusers build does not expose the required PixArt pipeline. "
                "Upgrade diffusers to a version with PixArt Alpha/Sigma support."
            )
        return "No compatible diffusers pipeline class was found for this model."

    def _load_pipeline_instance(
        self,
        pipeline_cls: type[Any],
        local_ref: LocalModelRef,
        torch_dtype: Any,
    ) -> Any:
        if local_ref.source_kind == "local_file":
            safetensor_files = sorted(local_ref.cache_path.glob("*.safetensors"))
            if len(safetensor_files) != 1:
                raise ProviderError(
                    "Expected exactly one .safetensors file in "
                    f"'{local_ref.cache_path}' for single-file loading."
                )
            single_file = safetensor_files[0]
            try:
                return pipeline_cls.from_single_file(single_file, torch_dtype=torch_dtype)
            except AttributeError as exc:
                raise ProviderError(
                    f"Pipeline '{pipeline_cls.__name__}' does not support "
                    f"from_single_file() for '{single_file.name}'."
                ) from exc
            except Exception as exc:
                raise ProviderError(
                    f"Failed to load single-file pipeline from '{single_file}': {exc}"
                ) from exc

        try:
            return pipeline_cls.from_pretrained(
                local_ref.cache_path,
                torch_dtype=torch_dtype,
                local_files_only=True,
                use_safetensors=True,
            )
        except TypeError:
            return pipeline_cls.from_pretrained(
                local_ref.cache_path,
                torch_dtype=torch_dtype,
                local_files_only=True,
            )
        except Exception as exc:
            raise ProviderError(
                f"Failed to load pipeline from '{local_ref.cache_path}': {exc}"
            ) from exc

    def _stage_local_source(
        self,
        local_source: Path,
        model_spec: ModelSpec,
        cache: CacheManager,
        ref: LocalModelRef,
        revision: str,
    ) -> LocalModelRef:
        staging_dir = cache.create_staging_dir(model_spec.canonical_id, revision)
        try:
            if local_source.is_dir():
                for child in local_source.iterdir():
                    target = staging_dir / child.name
                    if child.is_dir():
                        shutil.copytree(child, target)
                    else:
                        shutil.copy2(child, target)
            else:
                shutil.copy2(local_source, staging_dir / local_source.name)
            stats = cache.collect_file_stats(staging_dir)
            cache.finalize_download(
                staging_dir,
                ref,
                metadata={
                    "source": str(local_source),
                    "provider": model_spec.provider,
                    "revision": revision,
                    "source_kind": ref.source_kind,
                },
            )
            cache.append_registry_entry(
                {
                    "canonical_id": model_spec.canonical_id,
                    "provider": model_spec.provider,
                    "source": str(local_source),
                    "revision": revision,
                    "cache_path": str(ref.cache_path),
                    "license_hint": model_spec.license_hint,
                    "source_kind": ref.source_kind,
                    **stats,
                }
            )
            return ref
        except Exception as exc:
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise ProviderError(
                f"Failed to stage local source '{local_source}' "
                f"for model '{model_spec.canonical_id}': {exc}"
            ) from exc

    def _resolve_device(self, torch: Any, requested_device: str) -> tuple[str, list[str]]:
        warnings: list[str] = []
        if requested_device == "mps":
            available = bool(getattr(torch.backends, "mps", None)) and (
                torch.backends.mps.is_available()
            )
            if available:
                return "mps", warnings
            if requested_device == DEFAULT_GENERATE_DEVICE:
                warnings.append("Default MPS preference unavailable; falling back to CPU.")
                return "cpu", warnings
            raise ProviderError(
                "MPS was requested explicitly but is not available on this machine."
            )
        if requested_device == "cuda":
            if torch.cuda.is_available():
                return "cuda", warnings
            raise ProviderError(
                "CUDA was requested explicitly but is not available on this machine."
            )
        if requested_device == "cpu":
            return "cpu", warnings

        if bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available():
            return "mps", warnings
        if torch.cuda.is_available():
            return "cuda", warnings
        warnings.append("Requested accelerator unavailable; falling back to CPU.")
        return "cpu", warnings

    def _resolve_dtype(
        self,
        torch: Any,
        requested_dtype: str,
        device: str,
    ) -> tuple[Any, str, list[str]]:
        warnings: list[str] = []
        mapping = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }
        if requested_dtype not in mapping:
            raise ProviderError(
                f"Unsupported dtype '{requested_dtype}'. Use one of: fp16, bf16, fp32."
            )
        if device == "cpu" and requested_dtype == "fp16":
            warnings.append("fp16 is not practical on CPU; using fp32 instead.")
            return torch.float32, "fp32", warnings
        if device == "mps" and requested_dtype == "fp16":
            warnings.append("fp16 on MPS may produce NaNs/black images; using fp32 instead.")
            return torch.float32, "fp32", warnings
        if device == "mps" and requested_dtype == "bf16":
            warnings.append("bf16 is not broadly supported on MPS; using fp32 instead.")
            return torch.float32, "fp32", warnings
        return mapping[requested_dtype], requested_dtype, warnings

    def _make_generator(self, seed: int | None, device: str, num_images: int) -> Any:
        if seed is None:
            return None
        try:
            import torch
        except ImportError:
            return None
        generator_device = "cuda" if device == "cuda" else "cpu"
        return [
            torch.Generator(device=generator_device).manual_seed(seed + index)
            for index in range(num_images)
        ]
