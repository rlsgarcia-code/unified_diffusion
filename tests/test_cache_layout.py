from pathlib import Path

from unified_diffusion.cache.manager import CacheManager, resolve_revision, sanitize_path_component


def test_cache_layout_is_deterministic(tmp_path: Path) -> None:
    cache = CacheManager(tmp_path)
    model_dir = cache.model_dir("sdxl.base", "main")

    assert model_dir == tmp_path / "models" / "sdxl.base" / "main"
    assert cache.log_path == tmp_path / "logs" / "registry.jsonl"
    assert cache.tmp_dir == tmp_path / "tmp"


def test_sanitize_path_component_replaces_unsafe_chars() -> None:
    assert sanitize_path_component("sdxl/base@main") == "sdxl-base-main"


def test_resolve_revision_prefers_requested_value() -> None:
    assert resolve_revision("rev-a", "main") == "rev-a"
    assert resolve_revision(None, "main") == "main"
    assert resolve_revision(None, None) == "main"
