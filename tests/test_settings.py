from pathlib import Path

from unified_diffusion.settings import get_settings


def test_settings_use_defaults(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("UDIFF_CACHE_DIR", raising=False)
    monkeypatch.delenv("UDIFF_OUTPUT_DIR", raising=False)
    monkeypatch.delenv("UDIFF_REGISTRY_PATH", raising=False)
    monkeypatch.delenv("UDIFF_MODELS_DIR", raising=False)
    monkeypatch.delenv("UDIFF_API_HOST", raising=False)
    monkeypatch.delenv("UDIFF_API_PORT", raising=False)

    settings = get_settings()

    assert settings.default_registry_path == tmp_path / "custom-models.json"
    assert settings.api_host == "0.0.0.0"
    assert settings.api_port == 8000


def test_settings_read_environment(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("UDIFF_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("UDIFF_OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("UDIFF_REGISTRY_PATH", str(tmp_path / "registry.json"))
    monkeypatch.setenv("UDIFF_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("UDIFF_API_HOST", "127.0.0.1")
    monkeypatch.setenv("UDIFF_API_PORT", "9000")

    settings = get_settings()

    assert settings.cache_dir == tmp_path / "cache"
    assert settings.output_dir == tmp_path / "outputs"
    assert settings.default_registry_path == tmp_path / "registry.json"
    assert settings.default_models_dir == tmp_path / "models"
    assert settings.api_host == "127.0.0.1"
    assert settings.api_port == 9000
