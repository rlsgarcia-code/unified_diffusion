from __future__ import annotations

import json
import os
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import mkdtemp

from unified_diffusion.errors import CacheError


def sanitize_path_component(value: str) -> str:
    sanitized = []
    for char in value:
        if char.isalnum() or char in {".", "_", "-"}:
            sanitized.append(char)
        else:
            sanitized.append("-")
    result = "".join(sanitized).strip(".-")
    return result or "default"


def resolve_revision(requested_revision: str | None, default_revision: str | None) -> str:
    return requested_revision or default_revision or "main"


@dataclass(frozen=True, slots=True)
class LocalModelRef:
    canonical_id: str
    revision: str
    cache_path: Path
    source: str
    provider: str
    pipeline_type: str
    source_kind: str
    ready_marker: Path


class CacheManager:
    def __init__(self, cache_dir: str | Path, log_path: str | Path | None = None) -> None:
        self.root = Path(cache_dir).expanduser().resolve()
        self.models_dir = self.root / "models"
        self.tmp_dir = self.root / "tmp"
        self.logs_dir = self.root / "logs"
        self.log_path = (
            Path(log_path).expanduser().resolve()
            if log_path
            else self.logs_dir / "registry.jsonl"
        )
        self.hf_home = self.root / ".hf"
        self.hf_hub_cache = self.hf_home / "hub"
        self.transformers_cache = self.hf_home / "transformers"
        self._ensure_layout()

    def _ensure_layout(self) -> None:
        for path in (
            self.root,
            self.models_dir,
            self.tmp_dir,
            self.logs_dir,
            self.hf_home,
            self.hf_hub_cache,
            self.transformers_cache,
        ):
            path.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def model_dir(self, canonical_id: str, revision: str) -> Path:
        return (
            self.models_dir
            / sanitize_path_component(canonical_id)
            / sanitize_path_component(revision)
        )

    def local_ref(
        self,
        canonical_id: str,
        revision: str,
        source: str,
        provider: str,
        pipeline_type: str,
        source_kind: str = "remote_repo",
    ) -> LocalModelRef:
        cache_path = self.model_dir(canonical_id, revision)
        return LocalModelRef(
            canonical_id=canonical_id,
            revision=revision,
            cache_path=cache_path,
            source=source,
            provider=provider,
            pipeline_type=pipeline_type,
            source_kind=source_kind,
            ready_marker=cache_path / ".udiff-ready.json",
        )

    def is_downloaded(self, ref: LocalModelRef) -> bool:
        return ref.cache_path.exists() and ref.ready_marker.exists()

    def create_staging_dir(self, canonical_id: str, revision: str) -> Path:
        prefix = f"{sanitize_path_component(canonical_id)}-{sanitize_path_component(revision)}-"
        path = Path(mkdtemp(prefix=prefix, dir=self.tmp_dir))
        return path

    def finalize_download(
        self,
        staging_dir: Path,
        ref: LocalModelRef,
        metadata: dict[str, object] | None = None,
    ) -> None:
        ref.cache_path.parent.mkdir(parents=True, exist_ok=True)
        if ref.cache_path.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
            return
        try:
            os.replace(staging_dir, ref.cache_path)
            ref.ready_marker.write_text(
                json.dumps(metadata or {}, ensure_ascii=True, sort_keys=True, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise CacheError(f"Failed to finalize cache at '{ref.cache_path}': {exc}") from exc

    def collect_file_stats(self, path: Path) -> dict[str, object]:
        total_size = 0
        file_count = 0
        for file_path in path.rglob("*"):
            if file_path.is_file():
                file_count += 1
                total_size += file_path.stat().st_size
        return {"total_size": total_size, "file_count": file_count, "etag": None, "checksums": None}

    def append_registry_entry(self, entry: dict[str, object]) -> None:
        payload = {"timestamp": datetime.now(timezone.utc).isoformat(), **entry}
        try:
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")
        except OSError as exc:
            raise CacheError(f"Unable to append registry log at '{self.log_path}': {exc}") from exc

    def list_cached_models(self) -> list[str]:
        if not self.models_dir.exists():
            return []
        return sorted(
            str(path.relative_to(self.models_dir))
            for path in self.models_dir.glob("*/*")
            if path.is_dir()
        )

    @contextmanager
    def huggingface_env(self) -> Iterator[None]:
        env_updates = {
            "HF_HOME": str(self.hf_home),
            "HF_HUB_CACHE": str(self.hf_hub_cache),
            "TRANSFORMERS_CACHE": str(self.transformers_cache),
        }
        previous = {key: os.environ.get(key) for key in env_updates}
        os.environ.update(env_updates)
        try:
            yield
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
