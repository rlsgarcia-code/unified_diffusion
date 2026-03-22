import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from unified_diffusion.providers.flux_provider import FluxProvider
from unified_diffusion.registry.models import get_model_spec


def test_flux_provider_handles_flux_model() -> None:
    provider = FluxProvider()
    spec = get_model_spec("flux.1-dev")

    assert provider.can_handle(spec) is True


def test_flux_provider_pipeline_candidates_include_flux_pipeline() -> None:
    provider = FluxProvider()

    candidates = provider._pipeline_candidates(Path("/tmp/models/flux.1-dev/main"))

    assert candidates[0] == "FluxPipeline"


def test_flux_provider_error_message_mentions_flux_extra() -> None:
    provider = FluxProvider()

    message = provider._unsupported_pipeline_message(Path("/tmp/models/flux.1-dev/main"))

    assert "unified_diffusion[flux]" in message
    assert "FluxPipeline" in message


def test_flux_provider_load_pipeline_uses_flux_pipeline(monkeypatch, tmp_path: Path) -> None:
    provider = FluxProvider()
    model_dir = tmp_path / "models" / "flux.1-dev" / "main"
    model_dir.mkdir(parents=True)
    (model_dir / "model_index.json").write_text('{"_class_name": "FluxPipeline"}', encoding="utf-8")

    class FakePipeline:
        def __init__(self) -> None:
            self.moved_to = None
            self.attention_slicing_enabled = False
            self.vae = SimpleNamespace(enable_tiling=self._enable_tiling)
            self.vae_tiling_enabled = False

        @classmethod
        def from_pretrained(
            cls,
            path,
            torch_dtype=None,
            local_files_only=None,
            use_safetensors=None,
        ):
            instance = cls()
            instance.loaded_from = Path(path)
            instance.torch_dtype = torch_dtype
            instance.local_files_only = local_files_only
            instance.use_safetensors = use_safetensors
            return instance

        def to(self, device):
            self.moved_to = device
            return self

        def enable_attention_slicing(self):
            self.attention_slicing_enabled = True

        def _enable_tiling(self):
            self.vae_tiling_enabled = True

    fake_diffusers = ModuleType("diffusers")
    fake_diffusers.FluxPipeline = FakePipeline
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)

    fake_torch = ModuleType("torch")
    fake_torch.float16 = "float16"
    fake_torch.bfloat16 = "bfloat16"
    fake_torch.float32 = "float32"
    fake_torch.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False))
    fake_torch.cuda = SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    local_ref = SimpleNamespace(
        cache_path=model_dir,
        canonical_id="flux.1-dev",
        pipeline_type="flux",
        source_kind="local_dir",
    )
    loaded = provider.load_pipeline(local_ref, device="cpu", dtype="fp16", optimizations={})

    assert loaded.device_used == "cpu"
    assert loaded.dtype_used == "fp32"
    assert loaded.pipeline.loaded_from == model_dir
    assert loaded.pipeline.moved_to == "cpu"
    assert loaded.pipeline.attention_slicing_enabled is True
    assert loaded.pipeline.vae_tiling_enabled is True
    assert any("fp16 is not practical on CPU" in warning for warning in loaded.metadata["warnings"])
