import sys
from types import SimpleNamespace

import pytest

from unified_diffusion.errors import ProviderError
from unified_diffusion.providers.diffusers_provider import DiffusersProvider


class _FakeMPSBackend:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


class _FakeCudaBackend:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


class _FakeTorch:
    float16 = "float16"
    bfloat16 = "bfloat16"
    float32 = "float32"

    def __init__(self, mps_available: bool, cuda_available: bool) -> None:
        self.backends = SimpleNamespace(mps=_FakeMPSBackend(mps_available))
        self.cuda = _FakeCudaBackend(cuda_available)


def test_resolve_device_falls_back_to_cpu_when_no_accelerator() -> None:
    provider = DiffusersProvider()
    torch = _FakeTorch(mps_available=False, cuda_available=False)

    device, warnings = provider._resolve_device(torch, "cpu")

    assert device == "cpu"
    assert warnings == []


def test_resolve_device_errors_when_mps_is_explicit_and_missing() -> None:
    provider = DiffusersProvider()
    torch = _FakeTorch(mps_available=False, cuda_available=False)

    with pytest.raises(ProviderError) as exc:
        provider._resolve_device(torch, "mps")

    assert "MPS was requested explicitly" in str(exc.value)


def test_resolve_dtype_promotes_fp16_on_cpu_to_fp32() -> None:
    provider = DiffusersProvider()
    torch = _FakeTorch(mps_available=False, cuda_available=False)

    dtype_obj, dtype_name, warnings = provider._resolve_dtype(torch, "fp16", "cpu")

    assert dtype_obj == "float32"
    assert dtype_name == "fp32"
    assert "using fp32 instead" in warnings[0]


def test_resolve_dtype_demotes_bf16_on_mps_to_fp16() -> None:
    provider = DiffusersProvider()
    torch = _FakeTorch(mps_available=True, cuda_available=False)

    dtype_obj, dtype_name, warnings = provider._resolve_dtype(torch, "bf16", "mps")

    assert dtype_obj == "float32"
    assert dtype_name == "fp32"
    assert "using fp32 instead" in warnings[0]


def test_resolve_dtype_promotes_fp16_on_mps_to_fp32() -> None:
    provider = DiffusersProvider()
    torch = _FakeTorch(mps_available=True, cuda_available=False)

    dtype_obj, dtype_name, warnings = provider._resolve_dtype(torch, "fp16", "mps")

    assert dtype_obj == "float32"
    assert dtype_name == "fp32"
    assert "using fp32 instead" in warnings[0]


def test_unsupported_pipeline_message_mentions_sd3_upgrade() -> None:
    provider = DiffusersProvider()

    message = provider._unsupported_pipeline_message(
        __import__("pathlib").Path("/tmp/models/sd3.medium/main")
    )

    assert "Stable Diffusion 3" in message


def test_load_pipeline_forces_vae_fp32_on_mps_fp16(monkeypatch, tmp_path) -> None:
    provider = DiffusersProvider()
    model_dir = tmp_path / "models" / "sdxl.base" / "main"
    model_dir.mkdir(parents=True)
    (model_dir / "model_index.json").write_text(
        '{"_class_name": "StableDiffusionXLPipeline"}',
        encoding="utf-8",
    )

    class FakeVAE:
        def __init__(self) -> None:
            self.dtype_calls = []
            self.tiling_enabled = False

        def to(self, dtype=None):
            self.dtype_calls.append(dtype)
            return self

        def enable_tiling(self):
            self.tiling_enabled = True

    class FakePipeline:
        def __init__(self) -> None:
            self.moved_to = None
            self.attention_slicing_enabled = False
            self.vae = FakeVAE()

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def to(self, device):
            self.moved_to = device
            return self

        def enable_attention_slicing(self):
            self.attention_slicing_enabled = True

    fake_diffusers = SimpleNamespace(StableDiffusionXLPipeline=FakePipeline)
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)

    fake_torch = _FakeTorch(mps_available=True, cuda_available=False)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    local_ref = SimpleNamespace(
        cache_path=model_dir,
        canonical_id="sdxl.base",
        pipeline_type="stable-diffusion-xl",
        source_kind="local_dir",
    )
    loaded = provider.load_pipeline(local_ref, device="mps", dtype="fp16", optimizations={})

    assert loaded.device_used == "mps"
    assert loaded.pipeline.moved_to == "mps"
    assert loaded.pipeline.attention_slicing_enabled is True
    assert loaded.pipeline.vae.tiling_enabled is True
    assert loaded.pipeline.vae.dtype_calls == []
    assert any(
        "fp16 on MPS may produce NaNs/black images" in warning
        for warning in loaded.metadata["warnings"]
    )
