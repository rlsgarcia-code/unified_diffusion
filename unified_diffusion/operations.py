from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

from unified_diffusion.registry.models import CUSTOM_REGISTRY_ENV
from unified_diffusion.settings import get_settings
from unified_diffusion.telemetry import log_event


def configure_registry_path(registry_path_value: str | None = None) -> None:
    if registry_path_value:
        os.environ[CUSTOM_REGISTRY_ENV] = str(Path(registry_path_value).expanduser().resolve())
        log_event("registry.configured", source="argument", path=os.environ[CUSTOM_REGISTRY_ENV])
        return

    if os.environ.get(CUSTOM_REGISTRY_ENV):
        log_event("registry.configured", source="environment", path=os.environ[CUSTOM_REGISTRY_ENV])
        return

    default_registry_path = get_settings().default_registry_path
    if default_registry_path.exists():
        os.environ[CUSTOM_REGISTRY_ENV] = str(default_registry_path.resolve())
        log_event("registry.configured", source="default", path=os.environ[CUSTOM_REGISTRY_ENV])


def load_registry_object(registry_path: Path) -> dict[str, object]:
    if not registry_path.exists():
        return {}
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Registry file must contain a JSON object: {registry_path}")
    return payload


def verify_local_file(source_path: Path, expected_sha256: str | None = None) -> dict[str, str]:
    if not source_path.exists():
        raise FileNotFoundError(f"Model file not found: {source_path}")
    if source_path.suffix.lower() != ".safetensors":
        raise ValueError(f"Expected a .safetensors file, got: {source_path.name}")

    sha256_actual = compute_sha256(source_path)
    payload = {
        "path": str(source_path),
        "file_name": source_path.name,
        "file_size_bytes": str(source_path.stat().st_size),
        "sha256": sha256_actual,
        "sha256_verified": "false",
    }
    if expected_sha256:
        expected_normalized = expected_sha256.strip().lower()
        if sha256_actual != expected_normalized:
            log_event(
                "file.verify_failed",
                path=str(source_path),
                expected_sha256=expected_normalized,
                actual_sha256=sha256_actual,
            )
            raise ValueError(
                "SHA-256 mismatch for "
                f"'{source_path.name}': expected {expected_normalized}, got {sha256_actual}"
            )
        payload["sha256_verified"] = "true"
    log_event(
        "file.verified",
        path=str(source_path),
        sha256=sha256_actual,
        sha256_verified=payload["sha256_verified"],
    )
    return payload


def best_usage_practices() -> list[str]:
    return [
        (
            "Use `udiff verify-file --path ... --sha256 ...` before "
            "registering a custom .safetensors file."
        ),
        (
            "Prefer the Civitai SHA-256 when available; it catches "
            "incomplete or corrupted downloads early."
        ),
        "Keep default downloads in ~/.cache/unified-diffusion instead of inside the repository.",
        (
            "Keep custom local weights outside the repo, ideally in "
            "~/models/civitai/<slug>/model.safetensors."
        ),
        (
            "Use `udiff register-local` to move and normalize custom "
            "files instead of copying them manually."
        ),
        (
            "Use `pipeline_type=stable-diffusion-xl` for SDXL 1.0 style "
            "checkpoints unless you have evidence it needs another family."
        ),
        (
            "If a built-in model source breaks upstream, edit "
            "unified_diffusion/registry/models.py instead of patching the cache."
        ),
        (
            "Use `udiff guided-run --emit command` when you want a "
            "reproducible command before running a long generation."
        ),
    ]


def register_local_model_entry(
    source_path: Path,
    registry_path: Path,
    models_dir: Path,
    model_slug: str,
    canonical_id: str,
    provider: str,
    pipeline_type: str,
    default_revision: str,
    license_hint: str,
    notes: str,
    expected_sha256: str | None = None,
) -> dict[str, str]:
    verification = verify_local_file(source_path=source_path, expected_sha256=expected_sha256)
    sha256_actual = verification["sha256"]

    target_dir = models_dir / model_slug
    target_path = target_dir / "model.safetensors"
    target_dir.mkdir(parents=True, exist_ok=True)

    same_file = source_path.resolve() == target_path.resolve() if target_path.exists() else False
    if not same_file:
        if target_path.exists():
            raise FileExistsError(f"Target model file already exists: {target_path}")
        shutil.move(str(source_path), str(target_path))

    registry = load_registry_object(registry_path)
    registry[canonical_id] = {
        "provider": provider,
        "source": str(target_path),
        "pipeline_type": pipeline_type,
        "default_revision": default_revision,
        "license_hint": license_hint,
        "notes": notes,
    }
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    log_event(
        "registry.local_model_registered",
        canonical_id=canonical_id,
        moved_to=str(target_path),
        registry_path=str(registry_path),
    )

    sample_command = (
        f"UNIFIED_DIFFUSION_REGISTRY_PATH={registry_path} "
        f"uv run udiff run --model {canonical_id} "
        f'--prompt "portrait photo of a person, studio lighting" --out {model_slug}.png'
    )
    return {
        "canonical_id": canonical_id,
        "provider": provider,
        "pipeline_type": pipeline_type,
        "moved_to": str(target_path),
        "registry_path": str(registry_path),
        "sample_command": sample_command,
        "sha256_verified": "true" if bool(expected_sha256) else "false",
        "sha256_actual": sha256_actual or "",
    }


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
