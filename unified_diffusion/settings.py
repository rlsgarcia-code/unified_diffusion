from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Settings:
    cache_dir: Path
    output_dir: Path
    default_registry_path: Path
    default_models_dir: Path
    api_host: str
    api_port: int


def get_settings(cwd: str | Path | None = None) -> Settings:
    current_dir = Path(cwd).expanduser().resolve() if cwd else Path.cwd().resolve()
    return Settings(
        cache_dir=Path(
            os.environ.get("UDIFF_CACHE_DIR", "~/.cache/unified-diffusion")
        ).expanduser(),
        output_dir=Path(os.environ.get("UDIFF_OUTPUT_DIR", "outputs")).expanduser(),
        default_registry_path=Path(
            os.environ.get("UDIFF_REGISTRY_PATH", str(current_dir / "custom-models.json"))
        ).expanduser(),
        default_models_dir=Path(
            os.environ.get("UDIFF_MODELS_DIR", "~/models/civitai")
        ).expanduser(),
        api_host=os.environ.get("UDIFF_API_HOST", "0.0.0.0"),
        api_port=int(os.environ.get("UDIFF_API_PORT", "8000")),
    )
