import json
from pathlib import Path

from unified_diffusion.cache.manager import CacheManager


def test_registry_logging_writes_jsonl(tmp_path: Path) -> None:
    cache = CacheManager(tmp_path)

    cache.append_registry_entry(
        {
            "canonical_id": "sdxl.base",
            "provider": "diffusers",
            "source": "stabilityai/stable-diffusion-xl-base-1.0",
            "revision": "main",
            "cache_path": str(tmp_path / "models" / "sdxl.base" / "main"),
            "total_size": 123,
            "checksums": None,
            "etag": None,
            "license_hint": "test",
        }
    )

    lines = cache.log_path.read_text(encoding="utf-8").splitlines()
    payload = json.loads(lines[0])

    assert payload["canonical_id"] == "sdxl.base"
    assert payload["provider"] == "diffusers"
    assert payload["license_hint"] == "test"
