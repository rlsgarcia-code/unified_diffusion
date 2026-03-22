import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from unified_diffusion.cache.manager import CacheManager
from unified_diffusion.providers.diffusers_provider import DiffusersProvider
from unified_diffusion.registry.models import ModelSpec


def test_ensure_downloaded_stages_local_directory(tmp_path: Path) -> None:
    source_dir = tmp_path / "source-model"
    source_dir.mkdir()
    (source_dir / "model_index.json").write_text(
        '{"_class_name": "StableDiffusionXLPipeline"}',
        encoding="utf-8",
    )
    (source_dir / "weights.safetensors").write_text("stub", encoding="utf-8")

    provider = DiffusersProvider()
    cache = CacheManager(tmp_path / "cache")
    spec = ModelSpec(
        canonical_id="custom.local.dir",
        provider="diffusers",
        source=str(source_dir),
        pipeline_type="stable-diffusion-xl",
    )

    ref = provider.ensure_downloaded(spec, cache, "main")

    assert ref.source_kind == "local_dir"
    assert (ref.cache_path / "model_index.json").exists()
    assert (ref.cache_path / "weights.safetensors").exists()


def test_load_pipeline_uses_from_single_file_for_local_safetensors(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cache_path = tmp_path / "models" / "custom.single" / "main"
    cache_path.mkdir(parents=True)
    single_file = cache_path / "model.safetensors"
    single_file.write_text("stub", encoding="utf-8")

    class FakePipeline:
        def __init__(self) -> None:
            self.moved_to = None
            self.attention_slicing_enabled = False
            self.vae = SimpleNamespace(enable_tiling=self._enable_tiling)
            self.vae_tiling_enabled = False

        @classmethod
        def from_single_file(cls, path, torch_dtype=None):
            instance = cls()
            instance.loaded_from = Path(path)
            instance.torch_dtype = torch_dtype
            return instance

        def to(self, device):
            self.moved_to = device
            return self

        def enable_attention_slicing(self):
            self.attention_slicing_enabled = True

        def _enable_tiling(self):
            self.vae_tiling_enabled = True

    fake_diffusers = ModuleType("diffusers")
    fake_diffusers.StableDiffusionXLPipeline = FakePipeline
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)

    fake_torch = ModuleType("torch")
    fake_torch.float16 = "float16"
    fake_torch.bfloat16 = "bfloat16"
    fake_torch.float32 = "float32"
    fake_torch.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False))
    fake_torch.cuda = SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    local_ref = SimpleNamespace(
        canonical_id="custom.single",
        pipeline_type="stable-diffusion-xl",
        cache_path=cache_path,
        source_kind="local_file",
    )
    loaded = DiffusersProvider().load_pipeline(
        local_ref,
        device="cpu",
        dtype="fp32",
        optimizations={},
    )

    assert loaded.pipeline.loaded_from == single_file
    assert loaded.pipeline.moved_to == "cpu"
    assert loaded.pipeline.attention_slicing_enabled is True
    assert loaded.pipeline.vae_tiling_enabled is True
