import json
from pathlib import Path

import pytest
from PIL import Image

from unified_diffusion import Diffusion, GenerateRequest
from unified_diffusion.errors import ModelNotFoundError, RegistryValidationError
from unified_diffusion.providers.base import BaseProvider, LoadedPipeline
from unified_diffusion.registry.models import get_model_spec


class FakeProvider(BaseProvider):
    provider_name = "diffusers"

    def can_handle(self, model_spec):
        return True

    def ensure_downloaded(self, model_spec, cache, revision):
        ref = cache.local_ref(
            model_spec.canonical_id,
            revision,
            model_spec.source,
            model_spec.provider,
            model_spec.pipeline_type,
        )
        ref.cache_path.mkdir(parents=True, exist_ok=True)
        ref.ready_marker.write_text("{}", encoding="utf-8")
        return ref

    def load_pipeline(self, local_ref, device, dtype, optimizations):
        return LoadedPipeline(
            pipeline=object(),
            device_used=device,
            dtype_used=dtype,
            metadata={"warnings": []},
        )

    def generate(self, pipeline, request):
        return [Image.new("RGB", (request.width, request.height), color="black")]


def test_list_models_contains_expected_ids(tmp_path: Path) -> None:
    engine = Diffusion(cache_dir=tmp_path)
    models = engine.list_models()

    assert "sdxl.base" in models
    assert "sd21.base" in models
    assert "playground.v25" in models
    assert "flux.1-dev" in models


def test_unknown_model_raises_helpful_error(tmp_path: Path) -> None:
    engine = Diffusion(cache_dir=tmp_path)

    with pytest.raises(ModelNotFoundError) as exc:
        engine.run(GenerateRequest(model="does.not.exist", prompt="x"))

    assert "list_models" in str(exc.value)


def test_run_returns_generate_result_without_network(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    engine = Diffusion(cache_dir=tmp_path, default_device="cpu", default_dtype="fp32")
    monkeypatch.setattr(engine, "_get_provider", lambda provider_name: FakeProvider())

    result = engine.run(GenerateRequest(model="sdxl.base", prompt="hello", width=64, height=64))

    assert result.provider_used == "diffusers"
    assert result.model_resolved == "sdxl.base@main"
    assert result.images[0].size == (64, 64)


def test_generate_request_defaults_are_shared() -> None:
    request = GenerateRequest()

    assert request.model == "sdxl.base"
    assert request.prompt == "a dramatic product photo, studio lighting, clean background"
    assert request.negative_prompt == "blurry, low quality"
    assert request.width == 1024
    assert request.height == 1024
    assert request.steps == 30
    assert request.guidance_scale == 6.5
    assert request.seed == 1234
    assert request.device == "mps"
    assert request.dtype == "fp16"


def test_custom_registry_adds_five_models(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    custom_registry_path = tmp_path / "custom-models.json"
    custom_registry_path.write_text(
        json.dumps(
            {
                "research.custom.one": {
                    "provider": "diffusers",
                    "source": "org/model-one",
                    "pipeline_type": "stable-diffusion-xl",
                },
                "research.custom.two": {
                    "provider": "diffusers",
                    "source": "org/model-two",
                    "pipeline_type": "stable-diffusion",
                },
                "research.custom.three": {
                    "provider": "diffusers",
                    "source": "org/model-three",
                    "pipeline_type": "pixart-alpha",
                },
                "research.custom.four": {
                    "provider": "flux",
                    "source": "org/model-four",
                    "pipeline_type": "flux",
                },
                "research.custom.five": {
                    "provider": "diffusers",
                    "source": "org/model-five",
                    "pipeline_type": "stable-diffusion-3",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("UNIFIED_DIFFUSION_REGISTRY_PATH", str(custom_registry_path))

    engine = Diffusion(cache_dir=tmp_path)
    models = engine.list_models()

    assert "research.custom.one" in models
    assert "research.custom.five" in models
    assert get_model_spec("research.custom.four").provider == "flux"


def test_missing_custom_registry_file_raises_helpful_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    missing_path = tmp_path / "missing.json"
    monkeypatch.setenv("UNIFIED_DIFFUSION_REGISTRY_PATH", str(missing_path))

    with pytest.raises(ModelNotFoundError) as exc:
        Diffusion(cache_dir=tmp_path).list_models()

    assert str(missing_path) in str(exc.value)


def test_invalid_custom_registry_provider_raises_validation_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    custom_registry_path = tmp_path / "custom-models.json"
    custom_registry_path.write_text(
        json.dumps(
            {
                "research.invalid": {
                    "provider": "broken-provider",
                    "source": "org/model-one",
                    "pipeline_type": "stable-diffusion-xl",
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("UNIFIED_DIFFUSION_REGISTRY_PATH", str(custom_registry_path))

    with pytest.raises(RegistryValidationError) as exc:
        Diffusion(cache_dir=tmp_path).list_models()

    assert "unsupported provider" in str(exc.value)


def test_invalid_custom_registry_pipeline_type_raises_validation_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    custom_registry_path = tmp_path / "custom-models.json"
    custom_registry_path.write_text(
        json.dumps(
            {
                "research.invalid": {
                    "provider": "diffusers",
                    "source": "org/model-one",
                    "pipeline_type": "unknown-pipeline",
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("UNIFIED_DIFFUSION_REGISTRY_PATH", str(custom_registry_path))

    with pytest.raises(RegistryValidationError) as exc:
        Diffusion(cache_dir=tmp_path).list_models()

    assert "unsupported pipeline_type" in str(exc.value)
